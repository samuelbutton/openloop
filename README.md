# openloop

Python project scaffold. Product functionality will follow in later tasks.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run pre-commit install
```

Python 3.12 is selected by `.python-version`. uv installs it if needed.

Run the checks:

```sh
uv run pre-commit run --all-files
uv run pytest
```

Source code lives in `src/openloop/`; tests live in `tests/`.
Commit `uv.lock` when changing dependencies so development and CI use the same versions.

## License

Licensed under [Apache-2.0](LICENSE).
