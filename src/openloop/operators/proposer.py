"""Ledger-only proposer context. Model outputs cannot assign evidence or units."""

from openloop.ledger import (
    ActivityKind,
    InvalidInputError,
    JSONValue,
    Ledger,
    Status,
    content_hash,
)
from openloop.ledger.validation import coerce_enum, plain
from openloop.loops import LoopSpec, Phase

from .models import (
    HypothesisCard,
    ModelReply,
    Operator,
    PredictedEffect,
    ProposalDraft,
    ProposalProvider,
    ProposalRequest,
    check_parents,
    prompt_digest,
    untrusted_field,
)
from .spend import CallBudget, SpendControl, SpendGate, SpendTrace


def _failure_field(reason: object) -> JSONValue:
    """Bound recorded failure text; only strings can be quoted to a model."""
    return (
        untrusted_field(reason) if isinstance(reason, str) and reason.strip() else None
    )


class Proposer:
    """Bind provider predictions to a frozen loop and selected public evidence."""

    def __init__(
        self,
        spec: LoopSpec,
        ledger: Ledger,
        provider: ProposalProvider,
        *,
        gate: SpendGate | None = None,
        budget: CallBudget | None = None,
        simulated_only: bool = False,
    ) -> None:
        """Require a spend gate and call budget unless the provider is scripted.

        simulated_only=True refuses any reply that is not labeled simulated and
        records it as a proposal failure.
        """
        self.spec = spec
        self._ledger = ledger
        self._provider = provider
        self._spend = SpendControl(gate, budget, simulated_only=simulated_only)

    async def propose(
        self,
        operator: Operator | str,
        *,
        baseline: str,
        evidence: tuple[str, ...] = (),
        parents: tuple[str, ...] = (),
    ) -> HypothesisCard:
        """Read selected ledger runs, call a provider, then record a bound card.

        parents names ledger run IDs. The coordinator supplies their candidate
        hashes, the metric, its units, and the baseline fidelity, never the model.
        """
        operator = coerce_enum(Operator, operator, "operator")
        references = tuple(dict.fromkeys((baseline, *evidence, *parents)))
        runs = tuple(self._ledger.get(reference) for reference in references)
        baseline_run = runs[0]
        if baseline_run.status is not Status.SUCCEEDED or baseline_run.result is None:
            raise InvalidInputError("Prediction requires a successful baseline")
        for run in runs:
            self.spec.validate_inputs(run.inputs)
            if (
                run.inputs.phase not in (Phase.SCREEN, Phase.CONFIRM)
                or run.reused_from is not None
            ):
                raise InvalidInputError(
                    "Proposer evidence excludes held-out data and cache aliases"
                )
            if run.inputs.environment_hash != baseline_run.inputs.environment_hash:
                raise InvalidInputError("Proposer evidence environments differ")
            if run.result is not None:
                metric = run.result.metrics.get(self.spec.metric_name)
                if (
                    metric is None
                    or metric.split != "validation"
                    or metric.unit != self.spec.metric_unit
                    or metric.direction != self.spec.direction
                ):
                    raise InvalidInputError(
                        "Proposer requires the declared validation metric"
                    )
        parent_hashes = tuple(
            self._ledger.get(reference).inputs.candidate_hash for reference in parents
        )
        check_parents(operator, parent_hashes)
        if operator is Operator.DEBUG and not any(
            run.status is Status.FAILED for run in runs
        ):
            raise InvalidInputError("Debug requires failed experiment evidence")
        context = plain(
            {
                "loop_hash": self.spec.hash,
                "mutable_keys": self.spec.mutable_keys,
                "baseline": baseline,
                "metric": {
                    "name": self.spec.metric_name,
                    "unit": self.spec.metric_unit,
                    "direction": self.spec.direction,
                    "fidelity": baseline_run.inputs.fidelity,
                },
                "parents": parent_hashes,
                "evidence": tuple(
                    {
                        "run_id": run.id,
                        "candidate_hash": run.inputs.candidate_hash,
                        "inputs": run.inputs,
                        "status": run.status,
                        "metric": run.result.metrics[self.spec.metric_name]
                        if run.result
                        else None,
                        "failure": _failure_field(run.events[-1].payload.get("reason"))
                        if run.status is Status.FAILED
                        else None,
                    }
                    for run in runs
                ),
            }
        )
        assert isinstance(context, dict)
        request = ProposalRequest(operator, context)
        prompt = request.prompt
        prompt_hash = prompt_digest(prompt)
        trace = SpendTrace()
        try:
            reply = await self._spend.call(
                "proposal", prompt, lambda: self._provider(request), trace
            )
            if not isinstance(reply, ModelReply) or not isinstance(
                reply.value, ProposalDraft
            ):
                raise InvalidInputError(
                    "Provider must return a typed ProposalDraft reply"
                )
            card = HypothesisCard(
                operator,
                self.spec.hash,
                baseline,
                references,
                parent_hashes,
                reply.value,
                PredictedEffect(
                    self.spec.metric_name,
                    self.spec.metric_unit,
                    self.spec.direction,
                    baseline_run.inputs.fidelity,
                    reply.value.predicted_delta,
                    reply.value.probability_of_improvement,
                    reply.value.predicted_interval,
                ),
            )
        except BaseException as error:
            self._ledger.record_activity(
                ActivityKind.PROPOSAL_FAILURE,
                content_hash(request),
                {
                    "request": request,
                    "prompt_hash": prompt_hash,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "spend": trace.note,
                },
            )
            raise
        self._ledger.record_activity(
            ActivityKind.HYPOTHESIS,
            card.hash,
            {
                "card": card,
                "request_hash": content_hash(request),
                "prompt_hash": prompt_hash,
                "reply": reply,
                "spend": trace.note,
            },
            key=f"hypothesis:{content_hash(request)}:{reply.response_id}",
        )
        return card
