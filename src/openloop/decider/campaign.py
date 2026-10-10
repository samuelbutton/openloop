"""Trusted, in-memory coordination; no candidate workers or held-out file access.

The trusted final evaluator is called once, after selection closes. A failure
consumes that permission too. This guard is not a durable record or a sandbox:
the host must isolate held-out data and persist campaign snapshots externally.
"""

from collections.abc import Awaitable, Callable
from statistics import fmean

from openloop.ledger import InvalidInputError, InvalidTransitionError
from openloop.ledger.validation import coerce_enum

from .models import (
    ConfirmationScope,
    Decision,
    DecisionStage,
    Pair,
    Protocol,
    Sample,
    StagePlan,
    Verdict,
)
from .policies import decide, validate_calibration
from .statistics import Calibration

type FinalEvaluator = Callable[[str, StagePlan], Awaitable[tuple[Pair, ...]]]


class Campaign:
    """Freeze noise, screen, confirm, close selection, then evaluate held-out.

    Repeated calls may extend an inconclusive ordered prefix. Completed stages
    cannot be revised. Confirmation scores never change the screen policy.

    FAMILY scope confirms in any order at alpha / family size. SCREENED scope
    refuses confirmation until every screen is resolved, then fixes the set once
    from screen evidence alone (optionally the top k by mean screen gain) and
    tests it at alpha / set size. No confirmation score exists at that moment.
    """

    def __init__(
        self, protocol: Protocol, screen_noise: Calibration, confirm_noise: Calibration
    ) -> None:
        validate_calibration(protocol, DecisionStage.SCREEN, screen_noise)
        validate_calibration(protocol, DecisionStage.CONFIRM, confirm_noise)
        self._protocol = protocol
        self._screen_noise = screen_noise
        self._confirm_noise = confirm_noise
        self._decisions: dict[tuple[str, DecisionStage], Decision] = {}
        self._evidence: dict[tuple[str, DecisionStage], tuple[Pair, ...]] = {}
        self._samples: dict[str, Sample] = {}
        self._attempts: dict[str, Sample] = {
            sample.attempt_id: sample
            for noise in (screen_noise, confirm_noise)
            for sample in noise.samples
        }
        self._confirmation_set: tuple[str, ...] | None = None
        self._closed = False
        self._selected: str | None = None
        self._final_consumed = False
        self._final: Decision | None = None

    @property
    def protocol(self) -> Protocol:
        return self._protocol

    @property
    def decisions(self) -> tuple[Decision, ...]:
        """Return a detached immutable snapshot of validation decisions."""
        return tuple(self._decisions.values())

    @property
    def confirmation_set(self) -> tuple[str, ...] | None:
        """The frozen screened set, or None before it is fixed or in FAMILY scope."""
        return self._confirmation_set

    @property
    def confirmation_alpha(self) -> float | None:
        """The frozen per-test threshold, or None while a SCREENED set is open."""
        if self.protocol.confirmation_scope is ConfirmationScope.FAMILY:
            return self.protocol.test_alpha(len(self.protocol.candidates))
        if self._confirmation_set is None:
            return None
        return self.protocol.test_alpha(len(self._confirmation_set))

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def selected(self) -> str | None:
        return self._selected

    @property
    def final_consumed(self) -> bool:
        return self._final_consumed

    @property
    def final(self) -> Decision | None:
        """Release only the complete final verdict, after selection closed."""
        return self._final

    def _validate_samples(self, pairs: tuple[Pair, ...]) -> None:
        for pair in pairs:
            for sample in (pair.baseline, pair.candidate):
                prior = self._samples.get(sample.inputs.hash)
                if prior is not None and prior != sample:
                    raise InvalidInputError(
                        "A frozen input cannot acquire new decision evidence"
                    )
                prior = self._attempts.get(sample.attempt_id)
                if prior is not None and prior != sample:
                    raise InvalidInputError(
                        "An attempt cannot provide conflicting evidence"
                    )

    def _remember_samples(self, pairs: tuple[Pair, ...]) -> None:
        for pair in pairs:
            for sample in (pair.baseline, pair.candidate):
                self._samples[sample.inputs.hash] = sample
                self._attempts[sample.attempt_id] = sample

    def _freeze_confirmation_set(self) -> None:
        """Fix the SCREENED set once, from resolved screens and no confirmation score"""
        if self._confirmation_set is not None:
            return
        screens = {
            member: self._decisions.get((member, DecisionStage.SCREEN))
            for member in self.protocol.candidates
        }
        if any(
            decision is None or decision.verdict is Verdict.INCONCLUSIVE
            for decision in screens.values()
        ):
            raise InvalidTransitionError(
                "Resolve every screen before confirming a screened set"
            )
        promoted = sorted(
            (
                member
                for member, decision in screens.items()
                if decision is not None and decision.verdict is Verdict.PROMOTE
            ),
            key=lambda member: (
                -fmean(
                    pair.gain for pair in self._evidence[member, DecisionStage.SCREEN]
                ),
                member,
            ),
        )
        limit = self.protocol.confirm_top_k
        kept = promoted if limit is None else promoted[:limit]
        for member in promoted[len(kept) :]:
            screen = screens[member]
            assert screen is not None
            self._decisions[member, DecisionStage.CONFIRM] = Decision(
                self.protocol.hash,
                member,
                DecisionStage.CONFIRM,
                Verdict.DISCARD,
                "Promoted at screen but outside the confirmation top-k; not tested",
                (),
                0,
                screen.median_gain,
            )
        self._confirmation_set = tuple(kept)

    def evaluate(
        self, stage: DecisionStage, candidate: str, pairs: tuple[Pair, ...]
    ) -> Decision:
        """Accept validation evidence in stage order, with no optional stopping."""
        stage = coerce_enum(DecisionStage, stage, "decision stage")
        if self._closed or stage is DecisionStage.HELD_OUT:
            raise InvalidTransitionError(
                "Selection is closed or held-out requires the final evaluator"
            )
        if candidate not in self.protocol.candidates:
            raise InvalidInputError("Candidate is outside the frozen comparison family")
        if (
            stage is DecisionStage.CONFIRM
            and self.protocol.confirmation_scope is ConfirmationScope.SCREENED
        ):
            self._freeze_confirmation_set()
        key = candidate, stage
        old = self._decisions.get(key)
        if old is not None and old.verdict is not Verdict.INCONCLUSIVE:
            raise InvalidTransitionError("A completed stage cannot be revised")
        pairs = tuple(pairs)
        previous = self._evidence.get(key, ())
        if pairs[: len(previous)] != previous:
            raise InvalidInputError("An evidence prefix cannot be replaced")
        if stage is DecisionStage.CONFIRM:
            screen = self._decisions.get((candidate, DecisionStage.SCREEN))
            if screen is None or screen.verdict is not Verdict.PROMOTE:
                raise InvalidTransitionError("Confirmation requires a promoted screen")
        self._validate_samples(pairs)
        noise = (
            self._screen_noise if stage is DecisionStage.SCREEN else self._confirm_noise
        )
        result = decide(
            self.protocol,
            stage,
            candidate,
            pairs,
            noise,
            test_alpha=(
                self.confirmation_alpha if stage is DecisionStage.CONFIRM else None
            ),
        )
        self._decisions[key] = result
        self._evidence[key] = pairs
        self._remember_samples(pairs)
        return result

    def close_selection(self, candidate: str | None = None) -> None:
        """Close selection once the family is resolved; None closes with no winner.

        A candidate must be confirmed. None requires that no candidate was.
        """
        if self._closed:
            raise InvalidTransitionError("Selection is already closed")
        if candidate is not None:
            winner = self._decisions.get((candidate, DecisionStage.CONFIRM))
            if winner is None or winner.verdict is not Verdict.PROMOTE:
                raise InvalidTransitionError(
                    "Only a confirmed candidate can be selected"
                )
        if self.protocol.confirmation_scope is ConfirmationScope.SCREENED:
            self._freeze_confirmation_set()
        for member in self.protocol.candidates:
            screen = self._decisions.get((member, DecisionStage.SCREEN))
            confirm = self._decisions.get((member, DecisionStage.CONFIRM))
            if (
                screen is None
                or screen.verdict is Verdict.INCONCLUSIVE
                or (
                    screen.verdict is Verdict.PROMOTE
                    and (confirm is None or confirm.verdict is Verdict.INCONCLUSIVE)
                )
            ):
                raise InvalidTransitionError(
                    "Resolve the frozen family before final selection"
                )
            if (
                candidate is None
                and confirm is not None
                and confirm.verdict is Verdict.PROMOTE
            ):
                raise InvalidTransitionError("Select the confirmed candidate")
        self._closed = True
        self._selected = candidate

    async def finalize(self, evaluator: FinalEvaluator) -> Decision:
        """Consume held-out permission before invoking the trusted batch evaluator."""
        if self._selected is None or self._final_consumed:
            raise InvalidTransitionError(
                "Held-out requires closed selection and unused permission"
            )
        self._final_consumed = True
        pairs = tuple(await evaluator(self._selected, self.protocol.held_out))
        if len(pairs) != len(self.protocol.held_out.seeds):
            raise InvalidInputError(
                "Final evaluation must return the whole held-out batch"
            )
        self._validate_samples(pairs)
        result = decide(
            self.protocol,
            DecisionStage.HELD_OUT,
            self._selected,
            pairs,
            self._confirm_noise,
        )
        self._final = result
        self._remember_samples(pairs)
        return result
