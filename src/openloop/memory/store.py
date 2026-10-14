"""Research memory v0. The trusted coordinator writes; reports are read-only.

Only ledger-backed, replayed final verification can become a finding. Text from
hypothesis cards is not established fact. Ledger owns all persistent writes.
"""

from openloop.decider import Campaign, CampaignJournal
from openloop.ledger import (
    Derivation,
    Ledger,
    NotFoundError,
    content_hash,
)
from openloop.ledger.validation import check_digest

from .findings import describe_change, observe, operator_links
from .verification import verify

NAMESPACE = "research_memory.v0"


class ResearchMemory:
    """Publish verified findings and query immutable reports with evidence links.

    This is a report interface, not proposer context. Released held-out results
    must not guide adaptive search on the same held-out split.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    async def report(self, campaign: Campaign) -> tuple[str, dict[str, object]]:
        """Verify recorded evidence and return the finding's subject and payload.

        Nothing is written. The result depends only on ledger records and the
        campaign, so any process that replays the same records gets the same one.
        """
        audit = await verify(self.ledger, campaign)
        final = campaign.final
        assert final is not None  # verify rejects campaigns without a verified final.
        protocol = campaign.protocol
        held_out = protocol.held_out
        observation = observe(
            final,
            held_out.metric,
            held_out.metric_name,
            held_out.template.fidelity,
        )
        samples = audit.samples
        return final.candidate_hash, {
            "protocol_hash": protocol.hash,
            "protocol": protocol,
            "loop_hash": held_out.template.loop_hash,
            "decision": final,
            "decision_hash": content_hash(final),
            "selection": sorted(
                campaign.decisions,
                key=lambda item: (item.candidate_hash, item.stage),
            ),
            "external_screens": campaign.screen_sources,
            "confirmation_set": campaign.confirmation_set,
            "confirmation_alpha": campaign.confirmation_alpha,
            "held_out_pairs": final.sample_count,
            "evidence_attempts": len({sample.attempt_id for sample in samples}),
            "simulated": any(
                sample.inputs.execution.get("simulated") is True for sample in samples
            ),
            "registration": {
                "record_hash": audit.registration_hash,
                "registered_at": audit.registered_at,
                "first_run_at": audit.first_run_at,
            },
            "change": describe_change(audit.final_pairs),
            "links": operator_links(self.ledger, final.candidate_hash, observation),
            "evidence": [
                {
                    "run_id": sample.reference,
                    "attempt_id": sample.attempt_id,
                    "input_hash": sample.inputs.hash,
                    "metric_name": sample.metric_name,
                    "metric": sample.metric,
                    "execution": sample.inputs.execution,
                }
                for sample in samples
            ],
        }

    async def write(self, campaign: Campaign) -> Derivation:
        """Verify recorded evidence, then atomically publish one idempotent report.

        The protocol must have been registered on this ledger before the first
        comparison run (see `CampaignJournal.register`).
        """
        subject, payload = await self.report(campaign)
        return self.ledger.record_derivation(NAMESPACE, subject, payload)

    async def write_from_journal(self, protocol_hash: str) -> Derivation:
        """Rebuild a campaign from its journal in this process, then `write` it.

        Use this after a coordinator crash. Restoring grants no new held-out
        permission, and the published finding is the one the original process
        would have written.
        """
        campaign = await CampaignJournal(self.ledger).restore(protocol_hash)
        return await self.write(campaign)

    def query(
        self,
        *,
        candidate_hash: str | None = None,
        loop_hash: str | None = None,
        protocol_hash: str | None = None,
        simulated: bool | None = None,
    ) -> tuple[Derivation, ...]:
        """Return all matching findings in publication order, with their scope."""
        for digest in (loop_hash, protocol_hash):
            if digest is not None:
                check_digest(digest)
        return tuple(
            record
            for record in self.ledger.derivations(
                NAMESPACE, subject_hash=candidate_hash
            )
            if (loop_hash is None or record.payload["loop_hash"] == loop_hash)
            and (
                protocol_hash is None
                or record.payload["protocol_hash"] == protocol_hash
            )
            and (simulated is None or record.payload["simulated"] is simulated)
        )

    def get(self, finding_hash: str) -> Derivation:
        """Resolve one finding by its content hash."""
        check_digest(finding_hash)
        for record in self.query():
            if record.hash == finding_hash:
                return record
        raise NotFoundError(f"Unknown research finding: {finding_hash}")
