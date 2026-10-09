"""Load the downloaded tabular benchmarks through their official APIs."""

import argparse
import gc
import importlib.util
import json
import os
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tabular"


def probe_nas201() -> dict:
    from nas_201_api import NASBench201API

    # The trusted official v1.1 artifact uses legacy pickle serialization.
    # Current PyTorch defaults to a restricted format that this old API predates.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    api = NASBench201API(DATA / "raw/NAS-Bench-201-v1_1-096897.pth", verbose=False)
    assert len(api) == len(api.evaluated_indexes) == 15625
    queries = []
    for index in (0, 7812, 15624):
        for budget in ("12", "200"):
            metrics = api.get_more_info(
                index, "cifar10-valid", hp=budget, is_random=False
            )
            accuracy = float(metrics["valid-accuracy"])
            assert np.isfinite(accuracy) and 0 <= accuracy <= 100
            queries.append(
                {
                    "architecture_index": index,
                    "epochs": int(budget),
                    "validation_accuracy_percent": accuracy,
                    "recorded_training_seconds": metrics["train-all-time"],
                }
            )
    return {
        "architectures": len(api),
        "queries": queries,
        "torch_version": version("torch"),
        "api_version": version("nas-bench-201"),
    }


def probe_hpob() -> dict:
    spec = importlib.util.spec_from_file_location(
        "hpob_handler", ROOT / "research/hpo-b/hpob_handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handler = module.HPOBHandler(root_dir=str(DATA / "hpob/hpob-data"), mode="v3")
    counts = {}
    for name, tasks in (
        ("train", handler.meta_train_data),
        ("validation", handler.meta_validation_data),
        ("test", handler.meta_test_data),
    ):
        evaluations = 0
        dataset_tasks = 0
        for search_space in tasks.values():
            for task in search_space.values():
                x, y = np.asarray(task["X"]), np.asarray(task["y"])
                assert x.ndim == 2 and y.shape == (len(x), 1) and len(x) > 0
                assert np.isfinite(x).all() and np.isfinite(y).all()
                evaluations += len(x)
                dataset_tasks += 1
        counts[name] = {
            "search_spaces": len(tasks),
            "tasks": dataset_tasks,
            "evaluations": evaluations,
        }
    space = sorted(handler.get_search_spaces())[0]
    dataset = sorted(handler.get_datasets(space))[0]
    task = handler.meta_test_data[space][dataset]

    class FirstPending:
        def observe_and_suggest(self, X_obs, y_obs, X_pen):
            return 0

    history = handler.evaluate(FirstPending(), space, dataset, "test0", n_trials=1)
    assert len(history) == 2 and history[1] >= history[0]
    result = {
        "mode": "v3",
        "splits": counts,
        "initialization_seeds": handler.get_seeds(),
        "smoke_query": {
            "search_space": space,
            "dataset": dataset,
            "dimensions": handler.get_search_space_dim(space),
            "configurations": len(task["X"]),
            "normalized_incumbent_history": [float(y) for y in history],
        },
    }
    del handler
    gc.collect()
    # Also load the augmented table delivered in the archive (used by v1).
    with (DATA / "hpob/hpob-data/meta-train-dataset-augmented.json").open() as file:
        augmented = json.load(file)
    result["augmented_train_search_spaces"] = len(augmented)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("nas201", "hpob"))
    args = parser.parse_args()
    start = time.perf_counter()
    result = probe_nas201() if args.benchmark == "nas201" else probe_hpob()
    result["elapsed_seconds"] = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    path = DATA / f"{args.benchmark}-load.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
