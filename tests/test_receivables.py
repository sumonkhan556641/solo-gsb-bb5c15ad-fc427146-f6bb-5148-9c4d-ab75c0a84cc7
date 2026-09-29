"""收款核销试算核心逻辑测试（仅依赖标准库 + pytest，金额以分整数断言）。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.receivables import (
    InvoiceInput,
    PaymentEvent,
    ReversalEvent,
    TrialRejected,
    run_trial,
)


def _inv(no: str, due: str, cents: int) -> InvoiceInput:
    return InvoiceInput(no, date.fromisoformat(due), cents)


def _pay(eid: str, cents: int) -> PaymentEvent:
    return PaymentEvent(eid, cents)


def _rev(eid: str, target: str) -> ReversalEvent:
    return ReversalEvent(eid, target)


def _lines(payment):
    return [(line.invoice_no, line.amount_cents) for line in payment.lines]


def test_payments_fill_by_due_date_then_invoice_no_and_split():
    # 发票号字典序不影响顺序：先按到期日，同日再按发票号
    payments, outcomes, net = run_trial(
        invoices=[
            _inv("INV-10", "2026-02-01", 5000),
            _inv("INV-2", "2026-01-01", 3000),
            _inv("INV-1", "2026-02-01", 2000),
        ],
        events=[_pay("E1", 6000), _pay("E2", 4000)],
    )
    # E1：INV-2（最早）3000 + INV-1（同日字典序先于 INV-10）2000 + INV-10 1000
    assert _lines(payments[0]) == [
        ("INV-2", 3000),
        ("INV-1", 2000),
        ("INV-10", 1000),
    ]
    assert _lines(payments[1]) == [("INV-10", 4000)]
    assert net == 10000
    assert {o.invoice_no: o.outstanding_cents for o in outcomes} == {
        "INV-2": 0,
        "INV-1": 0,
        "INV-10": 0,
    }
    # 结果发票按 (到期日, 发票号) 排序
    assert [o.invoice_no for o in outcomes] == ["INV-2", "INV-1", "INV-10"]


def test_fully_paid_invoice_is_skipped_by_later_payments():
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100), _inv("B", "2026-02-01", 100)],
        events=[_pay("E1", 100), _pay("E2", 50)],
    )
    assert _lines(payments[0]) == [("A", 100)]
    assert _lines(payments[1]) == [("B", 50)]
    assert net == 150
    outstanding = {o.invoice_no: o.outstanding_cents for o in outcomes}
    assert outstanding == {"A": 0, "B": 50}


def test_reversal_releases_only_target_payment_and_others_keep_allocation():
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100), _inv("B", "2026-02-01", 100)],
        events=[
            _pay("E1", 100),  # 全部核销 A
            _pay("E2", 100),  # 全部核销 B
            _rev("R1", "E1"),  # 只释放 A 上的 100，E2 留在 B 不动
            _pay("E3", 50),  # 使用释放额度核销 A
        ],
    )
    by_id = {p.event_id: p for p in payments}
    assert by_id["E1"].reversed is True
    assert by_id["E2"].reversed is False
    assert by_id["E3"].reversed is False
    assert _lines(by_id["E1"]) == [("A", 100)]
    assert _lines(by_id["E2"]) == [("B", 100)]
    assert _lines(by_id["E3"]) == [("A", 50)]
    outstanding = {o.invoice_no: o.outstanding_cents for o in outcomes}
    assert outstanding == {"A": 50, "B": 0}
    # 净核销只计未撤回的收款：E2 100 + E3 50 = 150
    assert net == 150


def test_reversal_of_split_payment_refunds_each_original_line():
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 60), _inv("B", "2026-02-01", 60)],
        events=[
            _pay("E1", 100),  # A 60 + B 40
            _pay("E2", 20),  # B 20（B 余 0）
            _rev("R1", "E1"),  # 退回 A 60、B 40；E2 的 B 20 不动
        ],
    )
    by_id = {p.event_id: p for p in payments}
    assert _lines(by_id["E1"]) == [("A", 60), ("B", 40)]
    outstanding = {o.invoice_no: o.outstanding_cents for o in outcomes}
    assert outstanding == {"A": 60, "B": 40}
    assert net == 20  # 仅剩 E2 的 20


def test_released_capacity_consumed_in_due_order_by_later_payment():
    payments, outcomes, _ = run_trial(
        invoices=[_inv("A", "2026-01-01", 100), _inv("B", "2026-02-01", 100)],
        events=[
            _pay("E1", 150),  # A 100 + B 50
            _rev("R1", "E1"),  # 释放 A 100 + B 50
            _pay("E2", 120),  # 重新按顺序：A 100 + B 20
        ],
    )
    assert _lines(payments[1]) == [("A", 100), ("B", 20)]
    outstanding = {o.invoice_no: o.outstanding_cents for o in outcomes}
    assert outstanding == {"A": 0, "B": 80}


def test_identical_replay_of_payment_counts_once():
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100)],
        events=[
            _pay("E1", 40),
            _pay("E1", 40),  # 内容完全一致：跳过
            _pay("E2", 30),
        ],
    )
    assert [p.event_id for p in payments] == ["E1", "E2"]
    assert _lines(payments[0]) == [("A", 40)]
    assert net == 70
    assert outcomes[0].outstanding_cents == 30


def test_identical_replay_of_reversal_counts_once():
    # 第二次完全一致的撤回重放被跳过（不视为“撤回已撤回”而拒绝）
    payments, _, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100)],
        events=[
            _pay("E1", 50),
            _rev("R1", "E1"),
            _rev("R1", "E1"),
        ],
    )
    assert payments[0].reversed is True
    assert net == 0


def test_conflicting_replay_payment_amount_rejects_batch():
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 1000)],
            events=[_pay("E1", 10), _pay("E1", 20)],
        )
    exc = exc_info.value
    assert exc.error == "conflicting_replay"
    assert exc.event_index == 1
    assert exc.event_id == "E1"
    assert exc.detail()["error"] == "conflicting_replay"


def test_conflicting_replay_payment_vs_reversal_rejects_batch():
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 1000)],
            events=[_pay("E1", 10), _rev("E1", "E1")],
        )
    assert exc_info.value.error == "conflicting_replay"


def test_conflicting_replay_reversal_target_rejects_batch():
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 1000)],
            events=[
                _pay("E1", 10),
                _pay("E2", 10),
                _rev("R1", "E1"),
                _rev("R1", "E2"),
            ],
        )
    exc = exc_info.value
    assert exc.error == "conflicting_replay"
    assert exc.event_index == 3


def test_reversal_target_never_seen_as_payment_rejects():
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 1000)],
            events=[_rev("R1", "GHOST")],
        )
    exc = exc_info.value
    assert exc.error == "reversal_target_missing"
    assert exc.context["target_event_id"] == "GHOST"


def test_reversal_target_is_a_reversal_event_id_rejects():
    # 撤回引用的是另一条撤回事件号（从未作为收款出现）
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 1000)],
            events=[
                _pay("E1", 10),
                _rev("R1", "E1"),
                _rev("R2", "R1"),
            ],
        )
    assert exc_info.value.error == "reversal_target_missing"
    assert exc_info.value.context["target_event_id"] == "R1"


def test_double_reversal_of_same_payment_rejects():
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 1000)],
            events=[
                _pay("E1", 10),
                _rev("R1", "E1"),
                _rev("R2", "E1"),  # 不同事件号的第二次撤回
            ],
        )
    exc = exc_info.value
    assert exc.error == "reversal_target_already_reversed"
    assert exc.event_id == "R2"
    assert exc.context["target_event_id"] == "E1"


def test_payment_equal_to_outstanding_is_allowed():
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100)],
        events=[_pay("E1", 100)],
    )
    assert _lines(payments[0]) == [("A", 100)]
    assert outcomes[0].outstanding_cents == 0
    assert net == 100


def test_payment_exceeding_outstanding_rejects_with_detail():
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(
            invoices=[_inv("A", "2026-01-01", 100), _inv("B", "2026-02-01", 50)],
            events=[_pay("E1", 100), _pay("E2", 51)],
        )
    exc = exc_info.value
    assert exc.error == "payment_exceeds_outstanding"
    assert exc.event_index == 1
    assert exc.context == {"payment_amount": 51, "total_outstanding": 50}


def test_overpayment_check_uses_capacity_after_reversals():
    # E1 撤回后释放 100；E2 恰好 100 允许，101 则拒绝
    invoices = [_inv("A", "2026-01-01", 100)]
    base_events = [_pay("E1", 100), _rev("R1", "E1")]
    payments, outcomes, net = run_trial(
        invoices=invoices, events=base_events + [_pay("E2", 100)]
    )
    assert net == 100 and outcomes[0].outstanding_cents == 0
    with pytest.raises(TrialRejected) as exc_info:
        run_trial(invoices=invoices, events=base_events + [_pay("E2", 101)])
    assert exc_info.value.error == "payment_exceeds_outstanding"


def test_partial_payment_then_reversal_then_repay_with_cent_precision():
    # 非整元金额，验证按分精确：A=10.05, B=0.10
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 1005), _inv("B", "2026-02-01", 10)],
        events=[
            _pay("E1", 1000),  # A 10.00，A 余 5
            _rev("R1", "E1"),  # 释放 1000
            _pay("E2", 1015),  # A 1005 + B 10，恰好结清 1015
        ],
    )
    assert _lines(payments[1]) == [("A", 1005), ("B", 10)]
    assert all(o.outstanding_cents == 0 for o in outcomes)
    assert net == 1015


def test_no_events_or_no_payments_returns_original_receivables():
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100)], events=[]
    )
    assert payments == []
    assert outcomes[0].outstanding_cents == 100
    assert net == 0


def test_reversed_payment_keeps_original_lines_for_audit():
    payments, _, _ = run_trial(
        invoices=[_inv("A", "2026-01-01", 60), _inv("B", "2026-02-01", 60)],
        events=[_pay("E1", 100), _rev("R1", "E1")],
    )
    e1 = payments[0]
    assert e1.reversed is True
    assert e1.amount_cents == 100
    assert _lines(e1) == [("A", 60), ("B", 40)]


def test_replay_reversed_payment_id_is_skipped_not_reapplied():
    # E1 收款后被撤回；同号同内容的重放出现在撤回之后，仍然只是跳过
    payments, outcomes, net = run_trial(
        invoices=[_inv("A", "2026-01-01", 100)],
        events=[
            _pay("E1", 50),
            _rev("R1", "E1"),
            _pay("E1", 50),
        ],
    )
    assert len(payments) == 1
    assert payments[0].reversed is True
    assert outcomes[0].outstanding_cents == 100
    assert net == 0


def test_same_due_date_orders_invoices_by_unicode_codepoint():
    _, outcomes, _ = run_trial(
        invoices=[
            _inv("INV-10", "2026-01-01", 1),
            _inv("INV-9", "2026-01-01", 1),
            _inv("INV-09", "2026-01-01", 1),
        ],
        events=[],
    )
    # '-' (45) < '0' (48) < '9' (57)：INV-09 < INV-10 < INV-9
    assert [o.invoice_no for o in outcomes] == ["INV-09", "INV-10", "INV-9"]
