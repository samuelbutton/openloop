"""Pure adaptation of the pinned train.py: fixed tokens, seeded updates, no evaluator.

Only declared constant settings change. The trusted worker evaluates the model.
"""

import ast
from collections.abc import Mapping
from enum import StrEnum

from openloop.ledger.identity import JSONValue, check_positive, freeze_object

from .models import ContractError


class TrainingSetting(StrEnum):
    DEPTH = "DEPTH"
    ASPECT_RATIO = "ASPECT_RATIO"
    HEAD_DIM = "HEAD_DIM"
    WINDOW_PATTERN = "WINDOW_PATTERN"
    TOTAL_BATCH_SIZE = "TOTAL_BATCH_SIZE"
    DEVICE_BATCH_SIZE = "DEVICE_BATCH_SIZE"
    EMBEDDING_LR = "EMBEDDING_LR"
    UNEMBEDDING_LR = "UNEMBEDDING_LR"
    MATRIX_LR = "MATRIX_LR"
    SCALAR_LR = "SCALAR_LR"
    WEIGHT_DECAY = "WEIGHT_DECAY"
    ADAM_BETAS = "ADAM_BETAS"
    WARMUP_RATIO = "WARMUP_RATIO"
    WARMDOWN_RATIO = "WARMDOWN_RATIO"
    FINAL_LR_FRAC = "FINAL_LR_FRAC"


def _constant(node: ast.expr) -> JSONValue:
    if isinstance(node, ast.Constant):
        value: object = node.value
        if isinstance(value, (int, float, str)):
            return value
    if isinstance(node, ast.Tuple):
        return tuple(_constant(value) for value in node.elts)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.right, ast.Constant)
        and type(node.left.value) is int
        and type(node.right.value) is int
        and 0 <= node.right.value <= 32
    ):
        return node.left.value**node.right.value
    raise ContractError("Unsupported baseline constant expression")


def baseline_config(source: str) -> Mapping[str, JSONValue]:
    settings: dict[str, JSONValue] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in TrainingSetting:
                settings[target.id] = _constant(node.value)
    if set(settings) != set(TrainingSetting):
        raise ContractError("Training source lacks the declared settings")
    return freeze_object(settings)


class _TokenTraining(ast.NodeTransformer):
    def __init__(
        self,
        config: Mapping[str, JSONValue],
        seed: int,
        token_budget: int,
        tokenizer_dir: str,
    ) -> None:
        self.config = config
        self.seed = seed
        self.token_budget = token_budget
        self.tokenizer_dir = tokenizer_dir
        self.seeds = 0
        self.progress = 0
        self.stops = 0
        self.startups = 0

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.config:
                node.value = ast.parse(repr(self.config[name]), mode="eval").body
            elif name == "STARTUP_EXCLUDE_STEPS":
                node.value = ast.Constant(0)
                self.startups += 1
            elif name == "progress":
                node.value = ast.parse(
                    f"min(step * TOTAL_BATCH_SIZE / {self.token_budget}, 1.0)",
                    mode="eval",
                ).body
                self.progress += 1
            elif name == "remaining":
                node.value = ast.parse(
                    f"max(0, {self.token_budget} - (step + 1) * TOTAL_BATCH_SIZE)",
                    mode="eval",
                ).body
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        name = ast.unparse(node.func)
        if name == "mx.random.seed":
            node.args = [ast.Constant(self.seed)]
            self.seeds += 1
        elif name == "Tokenizer.from_directory":
            node.keywords = [
                ast.keyword("tokenizer_dir", ast.Constant(self.tokenizer_dir))
            ]
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        if any(
            isinstance(child, ast.Name) and child.id == "TIME_BUDGET"
            for child in ast.walk(node.test)
        ):
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Break):
                raise ContractError("Unexpected time-budget stop rule")
            node.test = ast.parse(
                f"step * TOTAL_BATCH_SIZE >= {self.token_budget}",
                mode="eval",
            ).body
            self.stops += 1
        self.generic_visit(node)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:
        for part in node.values:
            if isinstance(part, ast.Constant) and part.value == "s    ":
                part.value = " tokens    "
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        if (
            isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "print"
            and any(
                isinstance(child, ast.Name) and child.id == "TIME_BUDGET"
                for child in ast.walk(node)
            )
        ):
            return ast.Expr(
                ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[ast.Constant(f"Token budget: {self.token_budget}")],
                    keywords=[],
                )
            )
        self.generic_visit(node)
        return node


def training_program(
    source: str,
    config: Mapping[str, JSONValue],
    *,
    seed: int,
    token_budget: int,
    tokenizer_dir: str,
) -> str:
    """Return training code with a token stop, token schedule, and declared seed."""
    check_positive(token_budget)
    if type(seed) is not int or seed < 0:
        raise ContractError("Seed must be a nonnegative integer")
    if set(config) != set(TrainingSetting):
        raise ContractError("Training settings must be complete and normalized")
    batch = config[TrainingSetting.TOTAL_BATCH_SIZE]
    if type(batch) is not int or batch < 1 or token_budget % batch:
        raise ContractError("Token budget must contain an exact number of batches")
    tree = ast.parse(source)
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
        raise ContractError("Training source lacks the final-evaluation boundary")
    tree.body = [
        node
        for node in tree.body[: cutoff + 1]
        if not isinstance(node, ast.ImportFrom) or node.module != "prepare"
    ]
    transform = _TokenTraining(config, seed, token_budget, tokenizer_dir)
    transform.visit(tree)
    if (transform.seeds, transform.progress, transform.stops, transform.startups) != (
        1,
        1,
        1,
        1,
    ):
        raise ContractError("Training source differs from the supported token adapter")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
