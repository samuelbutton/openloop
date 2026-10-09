"""Ledger orchestration for single-candidate episodes. Runners are trusted adapters.

This contract limits experiment work. It is not a sandbox or an API spend gate.
"""

from collections.abc import Mapping
from contextlib import suppress

from openloop.ledger import (
    InvalidTransitionError,
    JSONValue,
    Ledger,
    Stage,
    Status,
    content_hash,
)
from openloop.ledger.validation import freeze_object

from .models import (
    ContractError,
    LoopAction,
    LoopSpec,
    Observation,
    Runner,
    StepResult,
    Workload,
)


class ExperimentEnv:
    """One candidate, one work allowance, and immutable observations; no reset.

    The allowance is a per-instance limit on work this object will admit. It is
    held in memory only: it does not survive a restart, and the ledger does not
    record it. Durable campaign budgets are planned.
    """

    def __init__(
        self,
        workload: Workload,
        ledger: Ledger,
        *,
        runner: Runner,
        config: Mapping[str, JSONValue],
        environment_hash: str,
        budget: int,
        hypothesis: str = "",
        parents: tuple[str, ...] = (),
    ) -> None:
        if type(budget) is not int or budget < 1:
            raise ContractError("Episode budget must be a positive integer")
        self._workload = workload
        self._spec = workload.spec
        self._ledger = ledger
        self._runner = runner
        self._config = freeze_object(workload.normalize_config(config))
        self._environment_hash = environment_hash
        self._remaining = budget
        self._hypothesis = hypothesis
        self._parents = tuple(parents)
        for parent in self._parents:
            ledger.get(parent)
        self._candidate = self._spec.candidate_hash(self._config)
        self._evidence = self._parents
        self._busy = False
        self._requests: dict[str, tuple[str, StepResult]] = {}

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    def _observation(self) -> Observation:
        return Observation(
            self._candidate,
            self._evidence,
            self.spec.budget_unit,
            self._remaining,
            tuple(
                item.name
                for item in self.spec.fidelities
                if item.amount <= self._remaining
            ),
        )

    async def initial_observation(self) -> Observation:
        return self._observation()

    async def step(self, action: LoopAction) -> StepResult:
        """Admit one sample; deduct its work before awaiting the trusted runner.

        The check against the remaining allowance is conservative: it runs
        before the ledger is consulted, so an action whose result would be a
        free cache hit is still refused when it exceeds the allowance. Failures
        keep the deduction because their actual work is unknown. Reused
        completed results consume no additional work or statistical sample.
        """
        fingerprint = content_hash(action)
        if action.request_key is not None and action.request_key in self._requests:
            previous, result = self._requests[action.request_key]
            if previous != fingerprint:
                raise ContractError("Request key already has a different action")
            return result
        if self._busy:
            raise ContractError("An episode cannot execute concurrent steps")
        if self._workload.spec != self.spec:
            raise ContractError("Loop specification changed during an episode")
        inputs = self.spec.inputs(self._config, action, self._environment_hash)
        job = self._workload.build_job(inputs)
        if job.inputs != inputs:
            raise ContractError("Job construction changed the declared inputs")
        if inputs.budget_amount > self._remaining:
            raise ContractError("Action exceeds the remaining episode budget")
        run = self._ledger.submit(
            inputs,
            hypothesis=self._hypothesis,
            parents=self._parents,
            request_key=action.request_key,
            purpose=action.purpose,
            retry_of=action.retry_of,
        )
        self._evidence += (run.id,)
        executed = False
        work_units = 0
        if run.reused_from is None and run.status is Status.QUEUED:
            self._busy = True
            self._remaining -= inputs.budget_amount
            try:
                self._ledger.start_stage(run.id, Stage.EXECUTION)
                output = await self._runner(job)
                if (
                    self._workload.spec != self.spec
                    or self._workload.build_job(inputs) != job
                ):
                    raise ContractError("Frozen loop or job changed during execution")
                self._ledger.start_stage(run.id, Stage.EVALUATION)
                result = self._workload.evaluate(job, output)
                work = output.get("work_units")
                if type(work) is not int or work != inputs.budget_amount:
                    raise ContractError(
                        "Successful work must equal the declared fidelity"
                    )
                run = self._ledger.complete(run.id, result)
                executed = True
                work_units = work
            except BaseException as error:
                with suppress(InvalidTransitionError):
                    self._ledger.fail(run.id, f"{type(error).__name__}: {error}")
                raise
            finally:
                self._busy = False
        observation = self._observation()
        response = StepResult(
            observation,
            run.id,
            run.attempt_id,
            run.status,
            run.result,
            executed,
            work_units,
            not observation.permitted_fidelities,
        )
        if action.request_key is not None:
            self._requests[action.request_key] = (fingerprint, response)
        return response
