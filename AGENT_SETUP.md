# Hosted agents and account setup

Model choices as of October 8, 2026 are recorded in `config/models.toml`.
The provider documents these model IDs without dated snapshots. Record the
requested and returned model IDs on each future experiment; do not imply that
an alias guarantees immutable weights.

| Role | Model / agent | Reasoning | Standard input / output per million tokens |
| --- | --- | --- | --- |
| Proposer | OpenAI `gpt-6-astra` | high | $10 / $50 |
| Implementer | Codex CLI, OpenAI `gpt-6.1-sol` | medium | $2 / $10 |

The proposer performs hypothesis generation and interpretation. The implementer
uses the existing coding agent to make bounded changes, run a smoke check, and
return a reviewable patch, with at most two implementation retries. Both are
hosted so the Mac's GPU and unified memory remain available to training.
This is the starting selection; no task-specific model comparison has been run.
Use the same proposer model, reasoning settings, and budget across study arms.

Sol is five times cheaper per standard uncached input/output token than Astra.
Cache writes, long context, fast modes, and paid hosted tools have different
prices; the two simple rates above are not an upper bound on all request costs.
Future runs should use standard service, bounded contexts, explicit output
limits, and separately recorded tool costs. Log billed usage, reasoning tokens,
cached tokens, request IDs, and cost per experiment.

Sources: [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

## API spending controls

**Status: verified at $10/month.** The dedicated `openloop` project shows
**Limit enforced**, with $0 used out of its $10 monthly limit. The user selected
this amount on October 8, 2026. Both account logins are verified. The active
`openloop-research` key has model-list Read and Responses Write access, with
all other capability groups disabled. It expires December 7, 2026.
The secret is stored only in the ignored local `.env` file, with owner-only
permissions (`0600`), as `OPENAI_API_KEY` and `CODEX_API_KEY`; the project ID is
stored alongside it. No credential appears in the ledger or tracked files.
A model-list request authenticated successfully and confirmed availability of
`gpt-6-astra` and `gpt-6.1-sol`. No paid model smoke call has been run.

Use a dedicated OpenAI project named `openloop` and a project-scoped key for both
roles. In project Settings → Limits → Spend → Edit spend limit, set $10/month
and enable **Enforce a hard limit**. Verify the saved enforcement
state, not just the amount or a spend alert. This limits requests billed to the
project rather than unrelated projects or the user's ChatGPT subscription.

The provider's enforced cap is monthly and can have a small overrun while its
state propagates. It is not an exact lifetime budget. A strict project-total cap
also requires the future executor to reserve conservative worst-case costs
before dispatch, include concurrent calls and retries, retain reservations on
unknown outcomes, and stop admitting work when the remaining budget is
insufficient. That executor is not implemented by this account-setup task.
Do not mark the total cap verified merely because the monthly setting is saved.

Keep secrets outside Git. `.env` files are ignored; do not paste keys into chat,
store them in the public ledger, or reuse unrelated keys. Configure Codex's API
authentication in an isolated runtime profile so it bills to the capped project
and does not silently use the user's existing subscription login. The installed
Codex CLI supports selecting the implementation model with `--model`.

The current setup provisions credentials and verifies model access. It does not
yet implement the proposer/implementer executor or run a billed coding session.

[OpenAI spend-limit documentation](https://developers.openai.com/api/docs/guides/spend-limits).

## RunPod credit

**Status: verified at $10 prepaid credit.** The user completed account setup and
funding directly in the browser, then changed both requested amounts to $10.
The Billing page shows a $10 balance, Auto-Pay Disabled, and $0.000/hr current
spend. No Pod or billable storage was provisioned by this setup.

Keep auto-pay disabled. Verify the balance before future GPU work and record
compute/storage costs and any later authorized top-ups. Account setup does not
authorize starting Pods or provisioning billable storage.

RunPod credits are prepaid and non-refundable. Auto-pay reloads the account by
charging a saved card, so enabling it would turn the requested one-time purchase
into additional spending. A $10 balance is not a lifetime budget if the account
is topped up again. Record compute and storage costs in the ledger when the GPU
workload is introduced.

[RunPod billing documentation](https://docs.runpod.io/accounts-billing/billing).
