"""收款对发票核销的试算逻辑（按分精确计算）。

规则：
- 发票按 ``(到期日, 发票号)`` 升序排列（发票号按 Unicode 码位字典序），
  每笔收款按该顺序依次填补未结清金额，**单笔收款可拆分到多张发票**。
- 事件按输入顺序处理，分两类：
  - 收款（``payment``）：携带正金额（至多两位小数，内部转为整数分）；
  - 撤回（``reversal``）：携带先前某笔收款的事件号，只释放该笔收款
    **原先核销**的金额；已处理的其他收款**不重新分配**，其后的收款可
    使用释放出的额度。
- 事件号去重：相同事件号且内容完全一致的重放**只计一次**（跳过）；
  相同事件号但内容不同则**整批拒绝**。
- 撤回目标收款不存在（事件号从未作为收款出现）或已撤回，也**整批拒绝**。
- 收款金额**严格大于**处理时刻全部发票的未结总额时整批拒绝
  （恰好结清允许）；本模块不做任何“提交”，任何拒绝都不产生部分结果。

金额在进入本模块前由 HTTP 层校验并转换为整数分，内部全程整数运算，
杜绝浮点误差。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

__all__ = [
    "InvoiceInput",
    "PaymentEvent",
    "ReversalEvent",
    "WriteOffLine",
    "PaymentResult",
    "InvoiceOutcome",
    "TrialRejected",
    "run_trial",
]


@dataclass(frozen=True)
class InvoiceInput:
    invoice_no: str
    due_date: date
    receivable_cents: int  # 应收金额（分），须为正整数


@dataclass(frozen=True)
class PaymentEvent:
    """收款事件：``kind == "payment"``，金额单位为分。"""

    event_id: str
    amount_cents: int


@dataclass(frozen=True)
class ReversalEvent:
    """撤回事件：``kind == "reversal"``，引用先前收款事件号。"""

    event_id: str
    target_event_id: str


@dataclass(frozen=True)
class WriteOffLine:
    """一笔收款在某张发票上的一次核销。"""

    invoice_no: str
    due_date: date
    amount_cents: int


@dataclass(frozen=True)
class PaymentResult:
    event_id: str
    amount_cents: int  # 收款原始金额（分）
    lines: list[WriteOffLine]  # 按发票填补顺序排列
    reversed: bool = False  # 该收款是否随后被撤回


@dataclass(frozen=True)
class InvoiceOutcome:
    invoice_no: str
    due_date: date
    receivable_cents: int  # 应收金额（分）
    outstanding_cents: int  # 试算结束时未结金额（分）


class TrialRejected(Exception):
    """整批试算被拒绝：冲突重放、非法撤回或收款超出未结总额。"""

    def __init__(
        self,
        error: str,
        event_index: int,
        event_id: str,
        message: str,
        **context: object,
    ) -> None:
        self.error = error
        self.event_index = event_index
        self.event_id = event_id
        self.context = context
        super().__init__(f"{error} at event[{event_index}] {event_id!r}: {message}")

    def detail(self) -> dict:
        """供 HTTP 层使用的结构化错误信息（409）。"""
        return {
            "error": self.error,
            "event_index": self.event_index,
            "event_id": self.event_id,
            **self.context,
        }


def _replay_key(event: PaymentEvent | ReversalEvent) -> tuple[object, ...]:
    """事件的规范化内容标识：同事件号重放时按它判断“内容完全一致”。"""
    if isinstance(event, PaymentEvent):
        return ("payment", event.amount_cents)
    return ("reversal", event.target_event_id)


def run_trial(
    invoices: list[InvoiceInput],
    events: list[PaymentEvent | ReversalEvent],
) -> tuple[list[PaymentResult], list[InvoiceOutcome], int]:
    """执行核销试算。

    返回 ``(各有效收款核销明细, 各发票未结金额, 净核销总额（分）)``；
    净核销总额 = 全部未撤回收款的核销额之和 = 应收总额 − 未结总额。
    任一拒绝条件触发时抛 :class:`TrialRejected`，不返回任何部分结果。
    """
    # 发票核销顺序：到期日升序，同日发票号 Unicode 字典序升序
    ordered_invoices = sorted(
        invoices, key=lambda inv: (inv.due_date, inv.invoice_no)
    )
    outstanding = {inv.invoice_no: inv.receivable_cents for inv in ordered_invoices}

    seen: dict[str, tuple[object, ...]] = {}  # 事件号 -> 规范化内容
    payment_index: dict[str, int] = {}  # 收款事件号 -> results 下标
    payment_reversed: dict[str, bool] = {}  # 收款事件号 -> 是否已撤回
    # 收款每一行当前仍由该收款占用的金额（撤回前等于行金额，撤回后为 0）
    unapplied: dict[str, list[int]] = {}
    results: list[PaymentResult] = []

    for index, event in enumerate(events):
        key = _replay_key(event)
        previous = seen.get(event.event_id)
        if previous is not None:
            if previous != key:
                raise TrialRejected(
                    "conflicting_replay",
                    event_index=index,
                    event_id=event.event_id,
                    message=(
                        "event id was already used with different content; "
                        "the whole batch is rejected"
                    ),
                )
            # 内容完全一致的重放：只计一次，直接跳过
            continue
        seen[event.event_id] = key

        if isinstance(event, ReversalEvent):
            target = event.target_event_id
            if target not in payment_index:
                raise TrialRejected(
                    "reversal_target_missing",
                    event_index=index,
                    event_id=event.event_id,
                    message=(
                        f"reversal target {target!r} is not an existing "
                        "payment event; the whole batch is rejected"
                    ),
                    target_event_id=target,
                )
            if payment_reversed[target]:
                raise TrialRejected(
                    "reversal_target_already_reversed",
                    event_index=index,
                    event_id=event.event_id,
                    message=(
                        f"payment {target!r} has already been reversed; "
                        "the whole batch is rejected"
                    ),
                    target_event_id=target,
                )
            # 只释放目标收款原先核销的金额：按其原始核销行逐张发票退回，
            # 其他收款的核销保持不动、不重新分配
            target_idx = payment_index[target]
            original_lines = results[target_idx].lines
            for line, still_applied in zip(
                original_lines, unapplied[target], strict=True
            ):
                if still_applied:
                    outstanding[line.invoice_no] += still_applied
            unapplied[target] = [0 for _ in original_lines]
            payment_reversed[target] = True
            results[target_idx] = PaymentResult(
                event_id=target,
                amount_cents=results[target_idx].amount_cents,
                lines=original_lines,
                reversed=True,
            )
            continue

        # 收款：按 (到期日, 发票号) 顺序填补未结金额，可拆分到多张发票
        total_open = sum(outstanding.values())
        if event.amount_cents > total_open:
            raise TrialRejected(
                "payment_exceeds_outstanding",
                event_index=index,
                event_id=event.event_id,
                message=(
                    "payment amount exceeds total outstanding receivable; "
                    "the whole batch is rejected"
                ),
                payment_amount=event.amount_cents,
                total_outstanding=total_open,
            )
        need = event.amount_cents
        lines: list[WriteOffLine] = []
        for inv in ordered_invoices:
            if need == 0:
                break
            open_amount = outstanding[inv.invoice_no]
            if open_amount <= 0:
                continue
            take = min(need, open_amount)
            lines.append(
                WriteOffLine(
                    invoice_no=inv.invoice_no,
                    due_date=inv.due_date,
                    amount_cents=take,
                )
            )
            outstanding[inv.invoice_no] = open_amount - take
            need -= take

        payment_index[event.event_id] = len(results)
        payment_reversed[event.event_id] = False
        unapplied[event.event_id] = [line.amount_cents for line in lines]
        results.append(
            PaymentResult(
                event_id=event.event_id,
                amount_cents=event.amount_cents,
                lines=lines,
            )
        )

    outcomes = [
        InvoiceOutcome(
            invoice_no=inv.invoice_no,
            due_date=inv.due_date,
            receivable_cents=inv.receivable_cents,
            outstanding_cents=outstanding[inv.invoice_no],
        )
        for inv in ordered_invoices
    ]
    total_receivable = sum(
        (inv.receivable_cents for inv in ordered_invoices), 0
    )
    total_outstanding = sum(outstanding.values())
    net_applied = total_receivable - total_outstanding
    return results, outcomes, net_applied
