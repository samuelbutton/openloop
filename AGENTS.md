Do not commit anything or create PRs. Only code changes.

# Rules for coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you change anything.
Its rules are requirements, not suggestions.
These additional rules apply to agents.

1. **Run every check before you report completion.**
   Run `uv run pre-commit run --all-files`, `uv run pyright`, and `uv run pytest`.
   Report failures with their output.
   Never claim a check passed if you did not run it.
2. **Spend nothing.**
   Do not call paid model APIs, start RunPod pods, or start GPU training runs without explicit approval in the current session.
3. **Do not touch secrets.**
   Do not read, print, copy, or edit `.env` or any credential file.
4. **Do not change frozen evidence.**
   Do not change `data/*/manifest.json`, `data/*/source.json`, `research/artifacts/`, the local `ledger.json`, or any SQLite ledger file.
   Do not change a golden-hash literal to make a test pass.
5. **Stay in scope.**
   Change only what the task requires.
   Report other problems you find instead of fixing them silently.
6. **Ask before you diverge.**
   If the task conflicts with [DESIGN.md](DESIGN.md) or CONTRIBUTING.md, stop and explain the conflict.
7. **Report precisely.**
   List the files you changed.
   Identify each decision that the task did not specify.
   State what you did not verify.
