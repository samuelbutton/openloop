# Tabular benchmark bring-up

These are BUILDPLAN.md's T4 lookup-table workloads. NAS-Bench-201 records
architecture training outcomes; HPO-B records hyperparameter evaluations.
Raw downloads and extracted data are local and ignored by Git. The load probes,
checksums, source identities, and reports are retained.

## Verified on October 8, 2026

Both official APIs loaded successfully on the M4 Pro with Python 3.12.13.
NAS-Bench-201 exposed all 15,625 architectures and passed six validation-metric
queries at the two training budgets. The probe took 107.47 seconds and peaked
at 8.93 GB RSS. For architecture 0, mean CIFAR-10 validation accuracy was
63.364% at 12 epochs and 81.983% at 200 epochs.

HPO-B v3 passed shape and finite-value checks for all these tables:

| Meta-split | Search spaces | Dataset tasks | Recorded evaluations |
| --- | ---: | ---: | ---: |
| Train | 16 | 758 | 3,279,050 |
| Validation | 16 | 91 | 1,180,786 |
| Test | 16 | 86 | 1,888,080 |

All five initialization seeds loaded. A one-suggestion lookup on search space
4796, dataset 23 completed with a valid, nondecreasing normalized incumbent
history. The augmented-training JSON also loaded, with 176 search spaces.
The HPO-B probe took 11.87 seconds and peaked at 5.08 GB RSS.

Evidence: [NAS load report](nas201-load.json), [HPO-B load report](hpob-load.json),
[download/extraction manifest](manifest.json), and [download checksums](downloads.sha256).
The local openloop ledger records both load reports and their source manifest.

## NAS-Bench-201

The official recommended release is `NAS-Bench-201-v1_1-096897.pth`, downloaded
from the Google Drive file linked by the
[official README](https://github.com/D-X-Y/NAS-Bench-201/blob/7fa5a61464e454e96bdb90818f0d795d68d5728b/README.md).
It is 5,003,180,601 bytes. Its full MD5 is
`55e847143ce1f7c2d89b676f6b096897`, matching the published six-character suffix.
The full SHA-256 is recorded in `manifest.json` and `downloads.sha256`.
This release contains lookup metrics, not the separate model-weight archives.

The official API is installed as `nas-bench-201` in the optional `benchmarks`
dependency group. The probe initializes the full API, checks all 15,625
architectures are represented, and queries indices 0, 7812, and 15624 on
`cifar10-valid` at both 12- and 200-epoch budgets, averaging the recorded seeds.
Validation accuracy is in percent and is maximized. Recorded training times are
historical benchmark measurements, not this Mac's execution times.

The old release uses pickle serialization. Its official loader predates modern
PyTorch's `weights_only=True` default, so the probe explicitly enables legacy
loading for this verified official artifact with
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Upstream code is unchanged.

## HPO-B

The full table archive comes from the author-provided link in the
[official README](https://github.com/machinelearningnuremberg/HPO-B/blob/f16c0c52c27ba4e928dd3ad648231726cb06a9ab/README.md),
pinned at data-repository revision
`6b9b6f54aeaad010760bba8f16ca0f070adcd63c`.
Its 261,011,374 bytes match the upstream Git LFS SHA-256
`ab1c439e50ffea3d8a3f1e0ad7ee7f03cd3c12df05576de84ad7c3b58fbf3358`.

It contains the original meta-train, meta-validation, meta-test, augmented-train,
and five-seed initialization JSON files. These task splits are preserved.
The probe loads `HPOBHandler(mode="v3")`, checks every task's configuration and
response shapes and finite values, runs a one-suggestion discrete lookup through
its `evaluate` method, and also loads the augmented-training table. The handler
normalizes responses to [0, 1] for its incumbent history; larger is better.
This is an API smoke check, not an allocator performance result. Continuous
surrogate models are outside this tabular setup.

The unmodified handler is pinned at repository revision
`f16c0c52c27ba4e928dd3ad648231726cb06a9ab` in `research/hpo-b/`.
It imports XGBoost even in discrete mode. On this Mac its import requires the
Homebrew `libomp` runtime; `brew install libomp` supplies it.

## Reproduce

From the repository root:

```sh
uv sync --locked --group benchmarks
# On macOS, if the OpenMP runtime is missing:
brew install libomp

git clone https://github.com/machinelearningnuremberg/HPO-B.git research/hpo-b
git -C research/hpo-b checkout f16c0c52c27ba4e928dd3ad648231726cb06a9ab
mkdir -p data/tabular/raw data/tabular/hpob
uv run --locked --group benchmarks gdown 'https://drive.google.com/uc?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_' -O data/tabular/raw/NAS-Bench-201-v1_1-096897.pth
curl -fL 'https://media.githubusercontent.com/media/sebastianpinedaar/hpo-data/6b9b6f54aeaad010760bba8f16ca0f070adcd63c/hpob-data.zip' -o data/tabular/raw/hpob-data.zip
shasum -a 256 -c data/tabular/downloads.sha256
uv run --locked --group benchmarks python -m zipfile -e data/tabular/raw/hpob-data.zip data/tabular/hpob
uv run --locked --group benchmarks python research/probe_tabular.py nas201
uv run --locked --group benchmarks python research/probe_tabular.py hpob
```

Run the two probes sequentially; each API loads its tables into memory.
The reports are `nas201-load.json` and `hpob-load.json`. Dependency versions are
locked by `uv.lock`; the benchmark group is not part of the default install.

## Attribution

NAS-Bench-201: Xuanyi Dong and Yi Yang, *NAS-Bench-201: Extending the Scope of
Reproducible Neural Architecture Search*, ICLR 2020. Official API license: MIT.

HPO-B: Sebastian Pineda-Arango, Hadi S. Jomaa, Martin Wistuba, and Josif Grabocka,
*HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML*,
NeurIPS Datasets and Benchmarks 2021. Official repository license: MIT.
Underlying OpenML datasets have their own source licenses; this setup keeps the
downloaded benchmark tables local rather than redistributing them.
