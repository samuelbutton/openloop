"""Pure composition of the published M4 Pro training recipe for T1.

The source bytes stay separate from this adapter. No optional runtime is imported.
The optimizer and model implementations retain the published arithmetic.
"""

import ast
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


class _SharedTraining(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"_hw_info", "_hp_defaults", "PEAK_FLOPS", "mfu"}:
                return None
            if name == "DEPTH":
                node.value = ast.Constant(6)
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if ast.unparse(node.func) == "make_dataloader":
            node.keywords = [key for key in node.keywords if key.arg != "backend"]
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        # Make the published warmup timer compatible with the token adapter.
        if ast.unparse(node.test) == "step > 10":
            node.test = ast.parse("step >= STARTUP_EXCLUDE_STEPS", mode="eval").body
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "print":
            names = {
                child.id for child in ast.walk(node) if isinstance(child, ast.Name)
            }
            if names & {"_hw_info", "PEAK_FLOPS"}:
                return None
            if "mfu" in names:
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
    tree.body = [
        node
        for node in tree.body[: cutoff + 1]
        if not isinstance(node, ast.ImportFrom)
        or node.module not in {"backends", "backends.muon_mlx"}
    ]
    _SharedTraining().visit(tree)
    tree.body = (
        ast.parse(optimizer).body
        + ast.parse("STARTUP_EXCLUDE_STEPS = 0").body
        + tree.body
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
