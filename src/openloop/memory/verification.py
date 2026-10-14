"""Audit a closed campaign against ledger measurements before memory admission.

The trusted coordinator supplies the protocol and campaign. Outcome labels and
caller-supplied statistics are insufficient: all scores come from Ledger.get.
The protocol must have a registration record on this ledger that is older than
every comparison run. That orders records on one trusted ledger; it cannot
authenticate a dishonest evaluator, or a coordinator that controls the ledger.
"""

from dataclasses import dataclass
from datetime import datetime

from openloop.decider import (
    Calibration,
    Campaign,
    CampaignJournal,
    Decision,
    DecisionStage,
    Pair,
    Policy,
    Sample,
    StagePlan,
    Verdict,
)
from openloop.ledger import InvalidInputError, InvalidTransitionError, Ledger


@dataclass(frozen=True)
class Audit:
    """Replayed evidence and the facts that order it after protocol registration."""

    samples: tuple[Sample, ...]
    final_pairs: tuple[Pair, ...]
    registration_hash: str
    registered_at: str
    first_run_at: str


class EvidenceReader:
    """Resolve successful original attempts and retain the exact admitted scores."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.samples: dict[tuple[str, str], Sample] = {}
        self.created_at: dict[str, str] = {}

    def sample(self, reference: str, metric_name: str) -> Sample:
        key = reference, metric_name
        if key not in self.samples:
            run = self.ledger.get(reference)
            self.samples[key] = Sample.from_run(run, metric_name)
            self.created_at[reference] = run.created_at
        return self.samples[key]

    def calibration(self, original: Calibration) -> Calibration:
        pairs = tuple(
            (
                self.sample(left.reference, left.metric_name),
                self.sample(right.reference, right.metric_name),
            )
            for left, right in original.pairs
        )
        noise = Calibration(pairs, original.kind)
        if noise != original:
            raise InvalidInputError("Calibration differs from ledger evidence")
        return noise

    def pairs(self, decision: Decision, plan: StagePlan) -> tuple[Pair, ...]:
        references = decision.evidence[: 2 * decision.sample_count]
        if len(references) != 2 * decision.sample_count:
            raise InvalidInputError("Decision evidence is incomplete")
        samples = tuple(self.sample(ref, plan.metric_name) for ref in references)
        return tuple(
            Pair(left, right)
            for left, right in zip(samples[::2], samples[1::2], strict=True)
        )


async def verify(ledger: Ledger, campaign: Campaign) -> Audit:
    """Replay the whole closed family and final test using recorded scores only.

    No evaluator is invoked. Cache aliases, absent runs, failed runs, substituted
    scores and changed thresholds cause rejection before any memory write.
    """
    final = campaign.final
    if (
        not campaign.closed
        or not campaign.final_consumed
        or final is None
        or final.verdict is not Verdict.VERIFIED
        or final.stage is not DecisionStage.HELD_OUT
        or final.candidate_hash != campaign.selected
    ):
        raise InvalidTransitionError("Memory requires a closed, verified final test")
    reader = EvidenceReader(ledger)
    replay = Campaign(
        campaign.protocol,
        reader.calibration(campaign.screen_noise),
        reader.calibration(campaign.confirm_noise),
    )
    for decision in campaign.decisions:
        if decision.stage is not DecisionStage.SCREEN:
            continue
        if campaign.protocol.policy is Policy.ALLOCATOR:
            seeds, source = campaign.screen_sources[decision.candidate_hash]
            for reference in decision.evidence:
                sample = reader.sample(reference, campaign.protocol.screen.metric_name)
                if (
                    sample.inputs.seed not in seeds
                    or sample.metric.split != "validation"
                    or sample.inputs.phase != DecisionStage.SCREEN.value
                    or sample.inputs.candidate_hash != decision.candidate_hash
                    or sample.inputs.data_hash
                    != campaign.protocol.screen.template.data_hash
                    or sample.inputs.loop_hash
                    != campaign.protocol.screen.template.loop_hash
                ):
                    raise InvalidInputError("External screen evidence is out of scope")
            actual = replay.record_external_screen(
                decision.candidate_hash,
                promoted=decision.verdict is Verdict.PROMOTE,
                evidence=decision.evidence,
                seeds=seeds,
                source=source,
            )
        else:
            actual = replay.evaluate(
                decision.stage,
                decision.candidate_hash,
                reader.pairs(decision, campaign.protocol.screen),
            )
        if actual != decision:
            raise InvalidInputError("Screen differs from ledger evidence")
    for decision in campaign.decisions:
        if decision.stage is DecisionStage.CONFIRM and decision.sample_count:
            actual = replay.evaluate(
                decision.stage,
                decision.candidate_hash,
                reader.pairs(decision, campaign.protocol.confirm),
            )
            if actual != decision:
                raise InvalidInputError("Confirmation differs from ledger evidence")
    replay.close_selection(campaign.selected)
    expected = {(item.candidate_hash, item.stage): item for item in campaign.decisions}
    actual = {(item.candidate_hash, item.stage): item for item in replay.decisions}
    if expected != actual or replay.confirmation_set != campaign.confirmation_set:
        raise InvalidInputError("Selection differs from the complete frozen family")

    async def recorded_final(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        return reader.pairs(final, plan)

    if await replay.finalize(recorded_final) != final:
        raise InvalidInputError("Final decision differs from ledger evidence")
    registration = CampaignJournal(ledger).registration(campaign.protocol.hash)
    # Strict order: the protocol, including its noise seeds, is registered before
    # every run it uses. Calibration counts too: seeds chosen after seeing
    # baseline scores could bias the noise estimate.
    first = min(reader.created_at.values(), key=datetime.fromisoformat)
    if datetime.fromisoformat(registration.timestamp) >= datetime.fromisoformat(first):
        raise InvalidTransitionError(
            "Protocol registration is not earlier than every run it uses, "
            "including calibration; register before calibrating"
        )
    return Audit(
        tuple(reader.samples[key] for key in sorted(reader.samples)),
        reader.pairs(final, campaign.protocol.held_out),
        registration.hash,
        registration.timestamp,
        first,
    )
