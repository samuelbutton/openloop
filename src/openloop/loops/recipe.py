"""Pure composition of the published M4 Pro training recipe for T1.

The source bytes stay separate from this adapter. No optional runtime is imported.
The optimizer and model implementations retain the published arithmetic.
"""

import ast
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType

from .models import ContractError

SHARED_RECIPE = "m4-pro-v1"
RECIPE_REVISION = "1d8377683d9a5b1af14f5933e6f5130c54e94460"
SHARED_TRAIN_SHA256 = "1a94a62eb8f08bccd3ca57b018da812b2009d733e7f795122ecabadf6ba69c4d"
RECIPE_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "train_mlx.py.txt": (
            "f53ec5cfee3b768627b02387645f5dca3ba2fec51ae24bf5521854bcba821d52"
        ),
        "muon_mlx.py.txt": (
            "a941fdb060dac1d9a76f5c19bd72b4996e67e3f387a8ee35fc29b173e8d54d38"
        ),
    }
)

# Each rewrite must match exactly this many times in the pinned upstream source.
# A missing or surplus match means the source no longer has the shape the composer
# was written for, which would otherwise pass silently as an unmodified program.
EXPECTED_REWRITES: Mapping[str, int] = MappingProxyType(
    {
        "import:backends": 2,
        "drop:_hw_info": 1,
        "drop:_hp_defaults": 1,
        "drop:PEAK_FLOPS": 1,
        "drop:mfu": 1,
        "set:DEPTH": 1,
        "dataloader:backend": 1,
        "timer:step>10": 1,
        "print:drop-hardware": 2,
        "print:rewrite-mfu": 1,
    }
)


class _SharedTraining(ast.NodeTransformer):
    def __init__(self) -> None:
        self.matches: Counter[str] = Counter()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"_hw_info", "_hp_defaults", "PEAK_FLOPS", "mfu"}:
                self.matches[f"drop:{name}"] += 1
                return None
            if name == "DEPTH":
                self.matches["set:DEPTH"] += 1
                node.value = ast.Constant(6)
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if ast.unparse(node.func) == "make_dataloader":
            kept = [key for key in node.keywords if key.arg != "backend"]
            self.matches["dataloader:backend"] += len(node.keywords) - len(kept)
            node.keywords = kept
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        # Make the published warmup timer compatible with the token adapter.
        if ast.unparse(node.test) == "step > 10":
            self.matches["timer:step>10"] += 1
            node.test = ast.parse("step >= STARTUP_EXCLUDE_STEPS", mode="eval").body
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "print":
            names = {
                child.id for child in ast.walk(node) if isinstance(child, ast.Name)
            }
            if names & {"_hw_info", "PEAK_FLOPS"}:
                self.matches["print:drop-hardware"] += 1
                return None
            if "mfu" in names:
                self.matches["print:rewrite-mfu"] += 1
                # No guessed hardware FLOPS or seconds label for remaining tokens.
                node.value = ast.parse(
                    'print(f"step {step} | loss: {train_loss_f:.6f} | '
                    'tok/sec: {tok_per_sec} | remaining_tokens: {remaining}", '
                    "flush=True)",
                    mode="eval",
                ).body
        self.generic_visit(node)
        return node


def shared_training(train: str, optimizer: str) -> str:
    """Inline the optimizer; fix the recipe independently of detected hardware."""
    tree = ast.parse(train)
    cutoff = next(
        (
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "total_tokens"
                for target in node.targets
            )
        ),
        None,
    )
    if cutoff is None:
        raise ContractError("Published recipe lacks its training boundary")
    rewriter = _SharedTraining()
    kept: list[ast.stmt] = []
    for node in tree.body[: cutoff + 1]:
        if isinstance(node, ast.ImportFrom) and node.module in {
            "backends",
            "backends.muon_mlx",
        }:
            rewriter.matches["import:backends"] += 1
        else:
            kept.append(node)
    tree.body = kept
    rewriter.visit(tree)
    if dict(rewriter.matches) != dict(EXPECTED_REWRITES):
        raise ContractError(
            "Published recipe does not match the composer's rewrite rules: "
            f"expected {dict(EXPECTED_REWRITES)}, matched {dict(rewriter.matches)}"
        )
    tree.body = (
        ast.parse(optimizer).body
        + ast.parse("STARTUP_EXCLUDE_STEPS = 0").body
        + tree.body
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
