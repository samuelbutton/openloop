# Research memory v0

`openloop.memory.ResearchMemory` implements the first research memory plugin (N7).
It stores verified positive findings from ledger measurements.
It does not convert a hypothesis, implementation success, screen promotion,
confirmation promotion, or `verified` activity label into a finding.

## Admission

Call `await memory.write(campaign)` after the campaign closes selection and
completes a successful held-out test.
After a coordinator crash, call `await memory.write_from_journal(protocol_hash)`
in a new process (see [Campaign journal](#campaign-journal)).
The memory plugin performs these checks before it writes a record:

1. Load each calibration and comparison run through `Ledger.get`.
2. Require successful original attempts with the declared metric.
   Reject cache aliases and missing or failed runs.
3. Rebuild calibration from the recorded scores.
4. Replay every screen and confirmation in the frozen comparison family.
   Check the confirmation set and its multiple-test threshold.
5. Replay the final test from recorded held-out scores.
   Compare the complete result with the campaign decision.
6. Require a protocol registration record on the same ledger.
   Its ledger time must be strictly earlier than the `created_at` of every
   screen, confirmation, and held-out run that the finding uses.

Replay does not invoke an evaluator, rerun training, or read a data split.
Both statistical screen policies and allocator screens are supported.
External screen references must resolve to validation runs for the declared
candidate, loop, data snapshot, and declared selection seeds.

The trusted coordinator supplies the original protocol and campaign.
The trusted evaluator supplies ledger measurements.
This plugin detects substituted statistics and evidence that disagrees with the
ledger. It cannot prove that an evaluator measured the intended workload honestly.
The low-level `Ledger.record_derivation` API trusts its plugin caller.
Use `ResearchMemory.write` for research memory admission.

## Pre-registration

`CampaignJournal(ledger).register(protocol)` records the frozen protocol in the
`campaign_protocol.v0` namespace. Registering the same protocol again returns
the first record, so a repeat never moves the time later.
`Campaign(..., journal=CampaignJournal(ledger))` registers the protocol when it
is created, before it accepts any decision.

What the check proves: on this ledger, a record of the exact protocol (seeds,
family, metric, thresholds) was written before the first comparison run was
created. The ledger assigns both times.

What it does not prove: a coordinator that controls the ledger file can write
any records in any order. The check also cannot show that the protocol came
before the coordinator saw the data in some other way, or that a run's
`created_at` is the moment its work began.

Calibration timing is strict: the protocol must be registered before every run
it uses, including calibration runs.
The noise seeds are part of the protocol. A coordinator who chose them after
seeing baseline scores could bias the noise estimate, so calibration counts.
Use this order: `CampaignJournal(ledger).register(protocol)`, then run the
calibration, then create the `Campaign` and run the comparisons.
`CampaignJournal.open` refuses a journaled campaign whose calibration runs
predate the registration, so the mistake fails at creation, not at publication.
`ResearchMemory.write` applies the same rule to every campaign, journaled or not.
Each finding reports `registration.registered_at` and `registration.first_run_at`.

## Campaign journal

A campaign with a journal records, in order and before it applies each one:
creation (with the calibration run references), every decision that `evaluate`
returns, every external screen with its seeds and source, the confirmation-set
freeze, the selection close, the start of the held-out test, and the final
decision. Each entry is an append-only derivation in `campaign_journal.v0`.
Each entry names its index and the hash of its predecessor. The first entry
names the registration record. An edit, gap, reorder, or fork breaks the chain.

`await CampaignJournal(ledger).restore(protocol_hash)` rebuilds the campaign.
It runs the recorded transitions through the normal `Campaign` methods. It reads
every score from `Ledger.get` and decides again. A recorded decision that
differs from the new one is rejected. Recorded statistics are never trusted.

Held-out permission is single-use. The start of the held-out test is journaled
before the evaluator runs. A restored campaign with that entry has
`final_consumed` set, and restore never calls an evaluator. If the coordinator
died during the held-out test, the restored campaign has no final decision and
`write_from_journal` raises. Nothing grants a second test.
A second `Campaign` for a protocol that already has a journal is refused.

A journal is not a lock. One coordinator must own it. Two writers can fork the
chain, and restore then fails. A coordinator that never opts in has no durable
guard: it can create a fresh in-memory `Campaign` and test held-out again.
`ResearchMemory.write` then still requires only the registration record.

## What a finding says

`change` compares the recorded baseline inputs and the verified candidate inputs
of the held-out pairs. It lists configuration keys added, removed, and changed
(old and new value, top-level keys), and `code_changed`, which is true when the
source or dependency hash differs. Both hashes appear for each side.
Every pair must show the same difference apart from the seed.

`links` lists each operator card that has a `candidate_bound` activity for this
candidate. Each link has the card hash, the binding sequence, the
implementation success sequence and snapshot hash, and the
`hypothesis_outcome` sequence when one exists. It puts the card's recorded
`prediction` (delta and probability of improvement) next to the `observation`
(signed delta, interval, pair count). The list is empty if no card is bound.
Hypothesis title, rationale, and edit text never appear. A prediction is the
model's claim and the observation is the measurement; they do not agree because
both are present.

## Records and queries

Each report contains the complete protocol, validation decisions, final decision,
confirmation set, threshold, and evidence links.
The final decision supplies the median paired gain, interval, p-value, noise
estimate, and held-out pair count.
A positive gain means improvement, for either metric direction.
An absent interval endpoint remains unbounded.

`evidence_attempts` counts distinct attempts across the whole verification audit,
including calibration and other candidates in the selection family.
`held_out_pairs` counts only the final baseline/candidate pairs.
Repeated publication of identical evidence returns the same record and time.
It adds no evidence or independent samples.
Each evidence entry includes its run ID, attempt ID, input hash, metric, and
execution settings. Reports retain the loop, data, evaluator, environment,
fidelity, and seed scope through the protocol and input references.

The `simulated` flag is true if any admitted evidence has that execution marker.
Every execution marker is also retained. A false flag is not proof of a native
hardware run. T0 evidence remains labeled as simulated.

```python
from openloop.ledger import Ledger
from openloop.memory import ResearchMemory

# Use an explicitly migrated working ledger, never a frozen evidence file.
with Ledger("working.sqlite3", migrate=True) as ledger:
    memory = ResearchMemory(ledger)
    finding = await memory.write(campaign)  # trusted, completed Campaign
    same_finding = memory.get(finding.hash)
    findings = memory.query(protocol_hash=campaign.protocol.hash)
```

The query API returns immutable `Derivation` records in publication order.
It has no hidden result limit. Filters select candidate, protocol, loop, or
simulation status. People and agents can read JSON with the same filters:

```sh
uv run python -m openloop.memory working.sqlite3
uv run python -m openloop.memory working.sqlite3 --candidate-hash <sha256>
```

The CLI opens the ledger read-only. It does not migrate it.
On a ledger older than schema 4 it exits with an explanation and changes nothing.
Schema 4 adds one append-only `derivation` table through the public ledger API.
Existing version 2 and 3 ledgers require an explicit migration of a working copy:
copy the file and open the copy with `Ledger(path, migrate=True)`.
Never migrate frozen evidence.
Publication uses one SQLite transaction. A killed writer leaves a full report
or no report. Concurrent duplicate writes produce one report.

## Limits

V0 stores positive findings only. A failed significance test does not prove that
an idea has no effect. Discards, inconclusive tests, and implementation failures
remain in their existing ledger records. They are not verified negative findings.
Hypothesis text is not copied into the report as an established conclusion.

This interface reports released findings. It does not add them to proposer
context. Further adaptive search requires a new untouched held-out split or an
approved reuse protocol. The S3 memory ablation remains planned.

The [memory tests](../tests/memory/test_memory.py) use recorded T0 runs.
They cover both statistical policies, screened top-k selection, allocator
selection, forged statistics, missing runs, cache aliases, idempotency, queries,
and a read-only CLI. The [finding tests](../tests/memory/test_findings.py) cover
the change and links. The [registration tests](../tests/memory/test_registration.py)
cover unregistered and late protocols. The
[journal tests](../tests/decider/test_journal.py) cover restore and tampering.
The [crash tests](../tests/memory/test_crash.py) `SIGKILL` a coordinator after
the final decision and during the held-out test. The
[storage tests](../tests/ledger/test_derivations.py) cover explicit migrations,
content hashes, concurrent writes, and `SIGKILL`.
