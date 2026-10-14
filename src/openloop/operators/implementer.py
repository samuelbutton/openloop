"""Bounded repair through a network-disabled ContainerExecutor and public ledger.

The model supplies text replacements. Unreviewed programs run only in Docker.
Every retry uses the same pinned source. Smoke passes never establish idea merit.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from openloop.executors import (
    ArtifactRole,
    ContainerExecutor,
    Limits,
    ProcessJob,
    State,
)
from openloop.ledger import (
    MAX_IMPLEMENTATION_ATTEMPTS,
    Activity,
    ActivityKind,
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    canonical_json,
    content_hash,
)
from openloop.ledger.validation import check_digest, check_positive, check_text

from .models import (
    HypothesisCard,
    ImplementationProvider,
    ImplementationRequest,
    ModelReply,
    Patch,
    prompt_digest,
)
from .spend import CallBudget, SpendControl, SpendGate, SpendLimitError, SpendTrace


@dataclass(frozen=True)
class ImplementedCandidate:
    card_hash: str
    snapshot_hash: str
    success_sequence: int
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        check_digest(self.card_hash)
        check_digest(self.snapshot_hash)
        check_positive(self.success_sequence)
        object.__setattr__(self, "files", Patch(self.files).files)


class Implementer:
    """Apply provider edits only inside the existing sandbox and smoke contract."""

    def __init__(
        self,
        executor: ContainerExecutor,
        ledger: Ledger,
        provider: ImplementationProvider,
        *,
        base_files: Mapping[str, str],
        smoke: tuple[str, ...],
        limits: Limits | None = None,
        gate: SpendGate | None = None,
        budget: CallBudget | None = None,
        simulated_only: bool = False,
    ) -> None:
        """Require a spend gate and call budget unless the provider is scripted.

        simulated_only=True refuses any reply that is not labeled simulated and
        records it as an implementation failure.
        """
        self._spend = SpendControl(gate, budget, simulated_only=simulated_only)
        if not isinstance(executor, ContainerExecutor) or executor.network:
            raise InvalidInputError(
                "Implementation requires a network-disabled ContainerExecutor"
            )
        self._executor = executor
        self._ledger = ledger
        self._provider = provider
        self._base_files = Patch(base_files).files
        self._limits = limits or Limits()
        self._smoke = ProcessJob(smoke, limits=self._limits).argv
        self._worker = Path(__file__).with_name("smoke_worker.py").read_bytes()

    async def implement(self, card: HypothesisCard) -> ImplementedCandidate | None:
        """Return a smoke-checked snapshot, or None after three failed attempts.

        Admission is atomic across callers. An interrupted start remains pending
        until its owner reconciles it with `reconcile`. A card that already has
        three failures returns None without a provider call. No retry can exceed
        the shared ledger's three-attempt cap for this card.
        """
        self._require_card(card)
        failures = [
            event
            for event in self._ledger.activities(subject_hash=card.hash)
            if event.kind is ActivityKind.IMPLEMENTATION_FAILURE
        ]
        if len(failures) >= MAX_IMPLEMENTATION_ATTEMPTS:
            return None
        last_reason = failures[-1].payload.get("reason") if failures else None
        previous = last_reason if isinstance(last_reason, str) else None
        for _ in range(MAX_IMPLEMENTATION_ATTEMPTS):
            start = self._ledger.record_activity(
                ActivityKind.IMPLEMENTATION_STARTED,
                card.hash,
                {
                    "source_commit": self._executor.source.commit,
                    "image": self._executor.image,
                    "smoke": self._smoke,
                    "limits": self._limits,
                    "base_hash": content_hash(self._base_files),
                    "worker_hash": hashlib.sha256(self._worker).hexdigest(),
                },
            )
            number = start.payload["attempt"]
            assert isinstance(number, int)
            job_id: str | None = None
            trace = SpendTrace()
            prompt_hash: str | None = None
            try:
                request = ImplementationRequest(
                    card,
                    self._base_files,
                    self._executor.source.commit,
                    number,
                    previous,
                )
                prompt_hash = prompt_digest(request.prompt)
                reply = await self._spend.call(
                    "implementation",
                    request.prompt,
                    partial(self._provider, request),
                    trace,
                )
                if not isinstance(reply, ModelReply) or not isinstance(
                    reply.value, Patch
                ):
                    raise InvalidInputError("Provider must return a typed Patch reply")
                patch = reply.value
                if set(patch.files) - self._base_files.keys():
                    raise InvalidInputError("Patch changes an unapproved file")
                if all(
                    self._base_files[name] == text for name, text in patch.files.items()
                ):
                    raise InvalidInputError("Implementation makes no source change")
                manifest = canonical_json(
                    {
                        "base_files": self._base_files,
                        "files": patch.files,
                        "smoke": self._smoke,
                    }
                ).encode()
                if (
                    sum(len(text.encode()) for text in patch.files.values())
                    > self._limits.outputs_bytes
                ):
                    raise InvalidInputError("Candidate exceeds the output size limit")
                job = ProcessJob(
                    ("python", "/inputs/worker.py"),
                    limits=self._limits,
                    files={"worker.py": self._worker, "manifest.json": manifest},
                )
                job_id = await self._executor.submit(
                    job, key=f"implement:{card.hash}:{start.sequence}"
                )
                self._ledger.record_activity(
                    ActivityKind.IMPLEMENTATION_PROGRESS,
                    card.hash,
                    {
                        "executor_job": job_id,
                        "reply": reply,
                        "prompt_hash": prompt_hash,
                        "spend": trace.note,
                        "patch_hash": patch.hash,
                    },
                    parent_sequence=start.sequence,
                )
                snapshot = await self._executor.collect(job_id)
                if snapshot.state is not State.SUCCEEDED:
                    diagnostics = []
                    for artifact in snapshot.artifacts:
                        if artifact.role is ArtifactRole.LOG:
                            data = artifact.path.read_bytes()
                            if hashlib.sha256(data).hexdigest() != artifact.sha256:
                                raise InvalidInputError("Smoke log hash changed")
                            diagnostics.append(data[-4000:].decode(errors="replace"))
                    raise InvalidInputError(
                        f"Smoke {snapshot.state}: {snapshot.error or ''}\n"
                        + "\n".join(diagnostics)
                    )
                output_root = self._executor.root / job_id / "outputs"
                outputs = {
                    artifact.path.relative_to(output_root).as_posix(): artifact
                    for artifact in snapshot.artifacts
                    if artifact.role is ArtifactRole.OUTPUT
                }
                if set(outputs) != set(patch.files):
                    raise InvalidInputError("Smoke outputs do not match the patch")
                for name, artifact in outputs.items():
                    expected = patch.files[name].encode()
                    if (
                        artifact.path.is_symlink()
                        or artifact.path.read_bytes() != expected
                        or hashlib.sha256(expected).hexdigest() != artifact.sha256
                    ):
                        raise InvalidInputError(
                            "Smoke output differs from candidate source"
                        )
                digest = content_hash(
                    {
                        "candidate_version": 1,
                        "base_commit": self._executor.source.commit,
                        "files": patch.files,
                    }
                )
                success = self._ledger.record_activity(
                    ActivityKind.IMPLEMENTATION_SUCCESS,
                    card.hash,
                    {
                        "snapshot_hash": digest,
                        "patch": patch,
                        "reply": reply,
                        "prompt_hash": prompt_hash,
                        "spend": trace.note,
                        "executor_job": job_id,
                        "artifacts": {
                            name: artifact.sha256 for name, artifact in outputs.items()
                        },
                    },
                    parent_sequence=start.sequence,
                )
                return ImplementedCandidate(
                    card.hash, digest, success.sequence, patch.files
                )
            except BaseException as error:
                if job_id is not None:
                    await self._executor.cancel(job_id)
                previous = f"{type(error).__name__}: {error}"
                self._ledger.record_activity(
                    ActivityKind.IMPLEMENTATION_FAILURE,
                    card.hash,
                    {
                        "reason": previous,
                        "executor_job": job_id,
                        "prompt_hash": prompt_hash,
                        "spend": trace.note,
                    },
                    parent_sequence=start.sequence,
                )
                if not isinstance(error, Exception) or isinstance(
                    error, SpendLimitError
                ):
                    raise
                if number == MAX_IMPLEMENTATION_ATTEMPTS:
                    return None
        return None

    def reconcile(
        self, card: HypothesisCard, reason: str, *, executor_job: str | None = None
    ) -> Activity:
        """Close a crashed attempt with a failure linked to its unresolved start.

        The caller must first confirm that the executor job, if any, is gone.
        This is an implementation failure, never evidence against the idea.
        """
        check_text(reason)
        self._require_card(card)
        history = self._ledger.activities(subject_hash=card.hash)
        starts = [
            event
            for event in history
            if event.kind is ActivityKind.IMPLEMENTATION_STARTED
        ]
        ended = {
            event.parent_sequence
            for event in history
            if event.kind
            in (
                ActivityKind.IMPLEMENTATION_FAILURE,
                ActivityKind.IMPLEMENTATION_SUCCESS,
            )
        }
        if not starts or starts[-1].sequence in ended:
            raise InvalidTransitionError("No unresolved implementation attempt")
        return self._ledger.record_activity(
            ActivityKind.IMPLEMENTATION_FAILURE,
            card.hash,
            {"reason": reason, "executor_job": executor_job, "reconciled": True},
            parent_sequence=starts[-1].sequence,
        )

    def _require_card(self, card: HypothesisCard) -> None:
        hypothesis = next(
            (
                event
                for event in self._ledger.activities(subject_hash=card.hash)
                if event.kind is ActivityKind.HYPOTHESIS
            ),
            None,
        )
        if hypothesis is None or canonical_json(
            hypothesis.payload.get("card")
        ) != canonical_json(card):
            raise InvalidInputError(
                "Implementer requires the exact recorded hypothesis card"
            )
