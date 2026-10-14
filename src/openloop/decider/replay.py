"""Rebuild a campaign from journal entries by re-deciding from ledger scores.

Pure replay over `Ledger.get`; it writes nothing. Recorded statistics are only
compared with the recomputed ones, never trusted.
"""

from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import cast

from openloop.ledger import (
    Derivation,
    ExperimentInputs,
    InvalidInputError,
    Ledger,
    Metric,
)

from .campaign import Campaign
from .models import (
    ConfirmationScope,
    Decision,
    DecisionStage,
    NoiseKind,
    Pair,
    Policy,
    Protocol,
    Sample,
    StagePlan,
)
from .statistics import Calibration
from .transitions import TransitionKind

type Object = Mapping[str, object]


def _object(value: object, label: str) -> Object:
    if not isinstance(value, Mapping):
        raise InvalidInputError(f"Recorded {label} is malformed")
    return cast(Object, value)  # keys are JSON strings; checked by the callers


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise InvalidInputError(f"Recorded {label} is malformed")
    return cast(Sequence[object], value)  # elements are checked by the callers


def _plan(value: object) -> StagePlan:
    data = _object(value, "stage plan")
    try:
        return StagePlan(
            ExperimentInputs(**_object(data["template"], "inputs")),  # pyright: ignore[reportArgumentType]
            Metric(**_object(data["metric"], "metric")),  # pyright: ignore[reportArgumentType]
            tuple(_sequence(data["seeds"], "seeds")),  # pyright: ignore[reportArgumentType]
            metric_name=data["metric_name"],  # pyright: ignore[reportArgumentType]
        )
    except (KeyError, TypeError) as error:
        raise InvalidInputError("Recorded stage plan is malformed") from error


def protocol_from_record(value: object) -> Protocol:
    """Rebuild a protocol from its recorded JSON; callers must check its hash."""
    data = _object(value, "protocol")
    try:
        return Protocol(
            tuple(_sequence(data["candidates"], "candidates")),  # pyright: ignore[reportArgumentType]
            _plan(data["screen"]),
            _plan(data["confirm"]),
            _plan(data["held_out"]),
            tuple(_sequence(data["screen_noise_seeds"], "seeds")),  # pyright: ignore[reportArgumentType]
            tuple(_sequence(data["confirm_noise_seeds"], "seeds")),  # pyright: ignore[reportArgumentType]
            policy=Policy(data["policy"]),
            alpha=data["alpha"],  # pyright: ignore[reportArgumentType]
            minimum_effect=data["minimum_effect"],  # pyright: ignore[reportArgumentType]
            screen_margin_sigmas=data["screen_margin_sigmas"],  # pyright: ignore[reportArgumentType]
            confirmation_scope=ConfirmationScope(data["confirmation_scope"]),
            confirm_top_k=data["confirm_top_k"],  # pyright: ignore[reportArgumentType]
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, InvalidInputError):
            raise
        raise InvalidInputError("Recorded protocol is malformed") from error


def _decision(value: object) -> Decision:
    try:
        return Decision(**_object(value, "decision"))  # pyright: ignore[reportArgumentType]
    except TypeError as error:
        raise InvalidInputError("Recorded decision is malformed") from error


class _Unfinished(Exception):
    """Stand-in final evaluator failure: the recorded final never completed."""


class Replay:
    """Apply recorded transitions to a fresh campaign, one entry at a time."""

    def __init__(
        self, ledger: Ledger, protocol: Protocol, creation: Derivation
    ) -> None:
        self.ledger = ledger
        self.protocol = protocol
        self._samples: dict[tuple[str, str], Sample] = {}
        body = _object(creation.payload["body"], "creation")
        try:
            self.campaign = Campaign(
                protocol,
                self._calibration(body["screen_noise"]),
                self._calibration(body["confirm_noise"]),
            )
        except KeyError as error:
            raise InvalidInputError("Recorded creation is malformed") from error
        self.final_started = False
        self.final_recorded = False

    def _sample(self, reference: object, name: str) -> Sample:
        if not isinstance(reference, str):
            raise InvalidInputError("Recorded run reference is malformed")
        if (reference, name) not in self._samples:
            self._samples[reference, name] = Sample.from_run(
                self.ledger.get(reference), name
            )
        return self._samples[reference, name]

    def _calibration(self, value: object) -> Calibration:
        data = _object(value, "calibration")
        name = data["metric_name"]
        if not isinstance(name, str):
            raise InvalidInputError("Recorded calibration is malformed")
        pairs = tuple(
            (self._sample(left, name), self._sample(right, name))
            for left, right in (
                _sequence(pair, "calibration pair")
                for pair in _sequence(data["pairs"], "calibration pairs")
            )
        )
        return Calibration(pairs, NoiseKind(data["kind"]))

    def _pairs(self, decision: Decision, plan: StagePlan) -> tuple[Pair, ...]:
        references = decision.evidence[: 2 * decision.sample_count]
        if len(references) != 2 * decision.sample_count:
            raise InvalidInputError("Journal decision evidence is incomplete")
        found = tuple(self._sample(item, plan.metric_name) for item in references)
        return tuple(Pair(a, b) for a, b in zip(found[::2], found[1::2], strict=True))

    @staticmethod
    def _same(actual: Decision, recorded: Decision) -> None:
        if actual != recorded:
            raise InvalidInputError("Journal decision differs from ledger evidence")

    async def apply(self, entry: Derivation) -> None:
        kind = TransitionKind(entry.payload["kind"])
        body = _object(entry.payload["body"], "transition")
        if self.final_recorded or (
            self.final_started and kind is not TransitionKind.FINAL
        ):
            raise InvalidInputError("Journal has entries after the final test")
        try:
            await self._apply(kind, body)
        except KeyError as error:
            raise InvalidInputError("Recorded transition is malformed") from error

    async def _apply(self, kind: TransitionKind, body: Object) -> None:
        campaign, protocol = self.campaign, self.protocol
        match kind:
            case TransitionKind.DECISION:
                recorded = _decision(body["decision"])
                plan = {
                    DecisionStage.SCREEN: protocol.screen,
                    DecisionStage.CONFIRM: protocol.confirm,
                }.get(recorded.stage)
                if plan is None:
                    raise InvalidInputError("Held-out is not a journal decision")
                self._same(
                    campaign.evaluate(
                        recorded.stage,
                        recorded.candidate_hash,
                        self._pairs(recorded, plan),
                    ),
                    recorded,
                )
            case TransitionKind.EXTERNAL_SCREEN:
                recorded = _decision(body["decision"])
                promoted = body["promoted"]
                if not isinstance(promoted, bool):
                    raise InvalidInputError("Recorded promotion is malformed")
                self._same(
                    campaign.record_external_screen(
                        recorded.candidate_hash,
                        promoted=promoted,
                        evidence=_sequence(body["evidence"], "evidence"),  # pyright: ignore[reportArgumentType]
                        seeds=_sequence(body["seeds"], "seeds"),  # pyright: ignore[reportArgumentType]
                        source=body["source"],  # pyright: ignore[reportArgumentType]
                    ),
                    recorded,
                )
            case TransitionKind.CONFIRMATION_SET:
                members = tuple(_sequence(body["members"], "confirmation set"))
                if campaign.freeze_confirmation_set() != members:
                    raise InvalidInputError(
                        "Journal confirmation set differs from the screens"
                    )
            case TransitionKind.SELECTION_CLOSED:
                selected = body["selected"]
                if selected is not None and not isinstance(selected, str):
                    raise InvalidInputError("Recorded selection is malformed")
                campaign.close_selection(selected)
            case TransitionKind.FINAL_STARTED:
                if body["candidate_hash"] != campaign.selected:
                    raise InvalidInputError("Final start names another candidate")
                self.final_started = True
            case TransitionKind.FINAL:
                final = _decision(body["decision"])

                async def recorded_final(
                    candidate: str, plan: StagePlan
                ) -> tuple[Pair, ...]:
                    return self._pairs(final, plan)

                self._same(await campaign.finalize(recorded_final), final)
                self.final_recorded = True
            case TransitionKind.CREATED:
                raise InvalidInputError("Journal has a second creation entry")

    async def finish(self) -> Campaign:
        """Consume the permission of a final that started but never completed."""
        if self.final_started and not self.final_recorded:

            async def unfinished(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
                raise _Unfinished

            with suppress(_Unfinished):
                await self.campaign.finalize(unfinished)
        return self.campaign
