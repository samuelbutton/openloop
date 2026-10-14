"""CPU-only static smoke check for edited T1 training source.

Boundary: stdlib only, so the file runs unchanged as `python -c` inside the
network-disabled sandbox, which has no MLX, GPU, or openloop package. It parses
and compiles the edited file, then checks the structural anchors that the T1
token-budget adapter in `openloop.loops.training` requires. It trains nothing
and measures nothing; a pass says the file is well formed, never that it is good.
The trusted T1 `build_job` re-validates the source outside the sandbox.
"""

import ast
import sys
from collections.abc import Sequence

# Mirrors openloop.loops.training.TrainingSetting; a unit test pins the equality.
SETTINGS = (
    "DEPTH",
    "ASPECT_RATIO",
    "HEAD_DIM",
    "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE",
    "DEVICE_BATCH_SIZE",
    "EMBEDDING_LR",
    "UNEMBEDDING_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "WEIGHT_DECAY",
    "ADAM_BETAS",
    "WARMUP_RATIO",
    "WARMDOWN_RATIO",
    "FINAL_LR_FRAC",
)


class SmokeCheckError(Exception):
    """The edited source cannot satisfy the T1 adapter contract."""


def _is_constant(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str))
    if isinstance(node, ast.Tuple):
        return all(_is_constant(value) for value in node.elts)
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.right, ast.Constant)
        and type(node.left.value) is int
        and type(node.right.value) is int
        and 0 <= node.right.value <= 32
    )


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def check_source(source: str, filename: str = "train.py") -> None:
    """Raise SmokeCheckError unless source compiles and keeps the adapter anchors.

    Nothing is executed and no bytecode is written: compile() builds the code
    object in memory and discards it.
    """
    try:
        compile(source, filename, "exec", dont_inherit=True)
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise SmokeCheckError(f"{filename} does not compile: {error}") from error
    settings: dict[str, ast.expr] = {}
    cutoff: int | None = None
    for index, node in enumerate(tree.body):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if len(node.targets) == 1 and target.id in SETTINGS:
                settings[target.id] = node.value
            if target.id == "total_tokens" and cutoff is None:
                cutoff = index
    missing = sorted(set(SETTINGS) - settings.keys())
    if missing:
        raise SmokeCheckError(f"Missing module-level settings: {', '.join(missing)}")
    for name, value in settings.items():
        if not _is_constant(value):
            raise SmokeCheckError(f"{name} must be a constant expression")
    if cutoff is None:
        raise SmokeCheckError("Missing the total_tokens final-evaluation boundary")
    counts = {"mx.random.seed": 0, "progress": 0, "STARTUP_EXCLUDE_STEPS": 0}
    stops = 0
    for statement in tree.body[: cutoff + 1]:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == (
                "mx.random.seed"
            ):
                counts["mx.random.seed"] += 1
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("progress", "STARTUP_EXCLUDE_STEPS")
            ):
                counts[node.targets[0].id] += 1
            elif isinstance(node, ast.If) and "TIME_BUDGET" in _names(node.test):
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Break):
                    raise SmokeCheckError("Unexpected time-budget stop rule")
                stops += 1
    counts["time-budget stop"] = stops
    wrong = {name: count for name, count in counts.items() if count != 1}
    if wrong:
        raise SmokeCheckError(
            f"Each adapter anchor must occur exactly once before total_tokens: {wrong}"
        )


def main(argv: Sequence[str]) -> int:
    """Check each named file; print the first problem to stderr and exit 1."""
    if not argv:
        print("usage: smoke_checks.py FILE...", file=sys.stderr)
        return 2
    for path in argv:
        try:
            with open(path, encoding="utf-8") as handle:
                check_source(handle.read(), path)
        except (OSError, UnicodeDecodeError, SmokeCheckError) as error:
            print(f"T1 smoke check failed: {error}", file=sys.stderr)
            return 1
    print("T1 smoke check passed (static only; nothing was trained)")
    return 0


def t1_smoke_command(path: str = "train.py") -> tuple[str, ...]:
    """Smoke argv that ships this file's source through `python -c`.

    Pass it to Implementer(smoke=...). The source is inlined because the
    sandbox accepts no extra input files and the repository is not importable
    there.
    """
    with open(__file__, encoding="utf-8") as handle:
        return ("python", "-c", handle.read(), path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
