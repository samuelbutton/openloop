"""Application spend gate for paid model calls, recorded in the public ledger.

Boundary: the caller supplies the dollar limit, per-model prices, and per-call
token budgets. Nothing here is hard-coded or fetched. A gate checks a single
coordinator process: the read of committed spend and the write of a reservation
are not one database transaction, so two processes sharing one ledger can
together pass a limit that either would respect. The provider account cap
remains the independent backstop.

Money is held in integer micro-USD so that sums and comparisons are exact.
Costs are rounded up. Unknown outcomes keep their full reservation.
"""

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from openloop.ledger import (
    Activity,
    ActivityKind,
    InvalidInputError,
    Ledger,
    LedgerError,
    content_hash,
)
from openloop.ledger.validation import JSONValue, check_text, finite_number

from .models import ModelReply, prompt_digest

MICRO_USD_PER_USD = 1_000_000
type Note = dict[str, JSONValue]


class SpendLimitError(LedgerError):
    """A paid call is refused, or a simulated-only operator got a paid reply."""


def _decimal(value: float, label: str) -> Fraction:
    number = finite_number(value, label)
    if number < 0:
        raise InvalidInputError(f"{label} must be nonnegative")
    # str(x) is the shortest decimal that round-trips, so 0.1 becomes exactly 1/10.
    return Fraction(str(number))


@dataclass(frozen=True)
class Pricing:
    """Caller-supplied USD per million tokens for one provider model alias.

    Prices are an input, never a constant. The record is hashed into every
    reservation so a later reader sees which prices were assumed.
    Cached input tokens are billed at the input price, a conservative choice.
    """

    provider: str
    model: str
    input_usd_per_million: float
    output_usd_per_million: float

    def __post_init__(self) -> None:
        check_text(self.provider)
        check_text(self.model)
        _decimal(self.input_usd_per_million, "Input price")
        _decimal(self.output_usd_per_million, "Output price")

    @property
    def hash(self) -> str:
        return content_hash({"pricing_version": 1, "pricing": self})

    def cost_micro_usd(self, input_tokens: int, output_tokens: int) -> int:
        """Round-up cost; USD per million tokens equals micro-USD per token."""
        return math.ceil(
            _decimal(self.input_usd_per_million, "Input price") * input_tokens
            + _decimal(self.output_usd_per_million, "Output price") * output_tokens
        )


@dataclass(frozen=True)
class CallBudget:
    """Declared upper bound for one call: its price and maximum token counts.

    max_input_tokens is checked against the prompt's UTF-8 byte length, an upper
    bound on token count, so no tokenizer is needed. The adapter must also
    enforce max_output_tokens at the provider.
    """

    pricing: Pricing
    max_input_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.pricing, Pricing):
            raise InvalidInputError("A call budget requires typed pricing")
        for count in (self.max_input_tokens, self.max_output_tokens):
            if type(count) is not int or count < 1:
                raise InvalidInputError("Token budgets must be positive integers")

    @property
    def max_cost_micro_usd(self) -> int:
        return self.pricing.cost_micro_usd(
            self.max_input_tokens, self.max_output_tokens
        )

    @property
    def hash(self) -> str:
        return content_hash({"call_budget_version": 1, "budget": self})


@dataclass(frozen=True)
class Reservation:
    sequence: int
    subject_hash: str
    max_cost_micro_usd: int
    budget: CallBudget


@dataclass(frozen=True)
class Committed:
    """Settled spend plus the full amount of every unsettled reservation."""

    settled_micro_usd: int
    open_micro_usd: int
    open_reservations: int

    @property
    def total_micro_usd(self) -> int:
        return self.settled_micro_usd + self.open_micro_usd


def _int_field(payload: Mapping[str, JSONValue], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise InvalidInputError(f"Spend record has an invalid {name}")
    return value


class SpendGate:
    """Reserve a conservative maximum before each paid call; settle afterwards."""

    def __init__(self, ledger: Ledger, *, limit_usd: float) -> None:
        self._ledger = ledger
        self.limit_micro_usd = math.floor(
            _decimal(limit_usd, "Spend limit") * MICRO_USD_PER_USD
        )
        if self.limit_micro_usd < 1:
            raise InvalidInputError("A spend limit must be positive")

    def committed(self) -> Committed:
        """Read settled and open amounts from every spend record in the ledger."""
        settled: dict[int, int] = {}
        reserved: dict[int, int] = {}
        for event in self._ledger.activities():
            if event.kind is ActivityKind.SPEND_RESERVATION:
                reserved[event.sequence] = _int_field(
                    event.payload, "max_cost_micro_usd"
                )
            elif event.kind is ActivityKind.SPEND_SETTLEMENT:
                if event.parent_sequence is None:
                    raise InvalidInputError("Settlement lost its reservation")
                settled[event.parent_sequence] = _int_field(
                    event.payload, "cost_micro_usd"
                )
        open_amounts = [
            amount for sequence, amount in reserved.items() if sequence not in settled
        ]
        return Committed(sum(settled.values()), sum(open_amounts), len(open_amounts))

    def reserve(self, budget: CallBudget, *, purpose: str, prompt: str) -> Reservation:
        """Refuse if settled + open + this maximum would exceed the limit.

        Every call, including a retry, needs its own reservation.
        """
        check_text(purpose)
        if len(prompt.encode()) > budget.max_input_tokens:
            raise SpendLimitError("Prompt exceeds the declared input token budget")
        maximum = budget.max_cost_micro_usd
        committed = self.committed()
        if committed.total_micro_usd + maximum > self.limit_micro_usd:
            raise SpendLimitError(
                "Spend limit would be exceeded: "
                f"{committed.settled_micro_usd} settled + "
                f"{committed.open_micro_usd} open + {maximum} requested > "
                f"{self.limit_micro_usd} micro-USD"
            )
        prompt_hash = prompt_digest(prompt)
        subject = content_hash(
            {"spend_call": prompt_hash, "purpose": purpose, "budget": budget.hash}
        )
        event = self._ledger.record_activity(
            ActivityKind.SPEND_RESERVATION,
            subject,
            {
                "purpose": purpose,
                "provider": budget.pricing.provider,
                "model": budget.pricing.model,
                "pricing": budget.pricing,
                "pricing_hash": budget.pricing.hash,
                "budget_hash": budget.hash,
                "max_input_tokens": budget.max_input_tokens,
                "max_output_tokens": budget.max_output_tokens,
                "max_cost_micro_usd": maximum,
                "limit_micro_usd": self.limit_micro_usd,
                "committed_before_micro_usd": committed.total_micro_usd,
                "prompt_hash": prompt_hash,
            },
        )
        return Reservation(event.sequence, subject, maximum, budget)

    def settle(self, reservation: Reservation, reply: ModelReply[object]) -> Note:
        """Settle from reported usage x pricing, or leave the reservation open.

        A simulated reply costs nothing. A reply with unreported usage, or from
        a provider other than the priced one, stays open at its full maximum.
        """
        pricing = reservation.budget.pricing
        if reply.simulated:
            return self._settle(reservation, reply, 0, "simulated")
        if reply.provider != pricing.provider:
            return self.hold(
                reservation,
                f"Reply provider {reply.provider!r} differs from priced "
                f"provider {pricing.provider!r}",
            )
        if reply.input_tokens is None or reply.output_tokens is None:
            return self.hold(reservation, "Provider did not report token usage")
        cost = pricing.cost_micro_usd(reply.input_tokens, reply.output_tokens)
        return self._settle(reservation, reply, cost, "reported_usage")

    def settle_reported(
        self, reservation: Reservation, *, cost_micro_usd: int, reason: str
    ) -> Activity:
        """Close an open reservation from an externally confirmed cost.

        The caller verified the amount in the provider's billing records.
        """
        check_text(reason)
        if type(cost_micro_usd) is not int or cost_micro_usd < 0:
            raise InvalidInputError("Cost must be a nonnegative integer")
        return self._ledger.record_activity(
            ActivityKind.SPEND_SETTLEMENT,
            reservation.subject_hash,
            {
                "basis": "reconciled",
                "reason": reason,
                "cost_micro_usd": cost_micro_usd,
                "reserved_micro_usd": reservation.max_cost_micro_usd,
            },
            parent_sequence=reservation.sequence,
        )

    def hold(self, reservation: Reservation, reason: str) -> Note:
        """Keep a reservation open. The returned note belongs in the caller's record."""
        return {
            "state": "open",
            "reservation_sequence": reservation.sequence,
            "reserved_micro_usd": reservation.max_cost_micro_usd,
            "why": reason,
        }

    def _settle(
        self,
        reservation: Reservation,
        reply: ModelReply[object],
        cost: int,
        basis: str,
    ) -> Note:
        pricing = reservation.budget.pricing
        event = self._ledger.record_activity(
            ActivityKind.SPEND_SETTLEMENT,
            reservation.subject_hash,
            {
                "basis": basis,
                "cost_micro_usd": cost,
                "reserved_micro_usd": reservation.max_cost_micro_usd,
                "exceeded_reservation": cost > reservation.max_cost_micro_usd,
                "provider": reply.provider,
                "model": reply.model,
                "configured_model": pricing.model,
                "response_id": reply.response_id,
                "pricing_hash": pricing.hash,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "reasoning_tokens": reply.reasoning_tokens,
                "cached_input_tokens": reply.cached_input_tokens,
            },
            parent_sequence=reservation.sequence,
        )
        return {
            "state": "settled",
            "reservation_sequence": reservation.sequence,
            "settlement_sequence": event.sequence,
            "cost_micro_usd": cost,
        }


@dataclass
class SpendTrace:
    """Latest spend state of one call, readable by the caller after an error."""

    note: Note | None = None


class SpendControl:
    """Bind an operator to either a spend gate or an explicit simulated-only mode."""

    def __init__(
        self,
        gate: SpendGate | None,
        budget: CallBudget | None,
        *,
        simulated_only: bool,
    ) -> None:
        if simulated_only:
            if gate is not None or budget is not None:
                raise InvalidInputError("A simulated-only operator takes no spend gate")
        elif not isinstance(gate, SpendGate) or not isinstance(budget, CallBudget):
            raise InvalidInputError(
                "A paid operator requires a SpendGate and a CallBudget; "
                "pass simulated_only=True for scripted providers"
            )
        self._gate = gate
        self._budget = budget
        self.simulated_only = simulated_only

    async def call[T](
        self,
        purpose: str,
        prompt: str,
        send: Callable[[], Awaitable[ModelReply[T]]],
        trace: SpendTrace,
    ) -> ModelReply[T]:
        """Run one provider call under the gate; trace.note reflects the outcome."""
        if self._gate is None or self._budget is None:
            reply = await send()
            if isinstance(reply, ModelReply) and not reply.simulated:
                trace.note = {
                    "state": "unreserved",
                    "why": "A non-simulated reply reached a simulated-only operator",
                }
                raise SpendLimitError(
                    "Simulated-only operator received a non-simulated reply; "
                    "the call was made without a reservation"
                )
            trace.note = {"state": "simulated_only"}
            return reply
        reservation = self._gate.reserve(self._budget, purpose=purpose, prompt=prompt)
        trace.note = self._gate.hold(reservation, "Provider call started")
        try:
            reply = await send()
        except BaseException as error:
            trace.note = self._gate.hold(
                reservation, f"Call ended without a reply: {type(error).__name__}"
            )
            raise
        if not isinstance(reply, ModelReply):
            trace.note = self._gate.hold(reservation, "Reply was not a ModelReply")
            return reply
        trace.note = self._gate.settle(reservation, reply)
        return reply
