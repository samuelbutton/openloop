"""Prepare or execute the Amendment 5 completion of the held-out design.

The Amendment 4 run (`run-heldout-2026-10-13c`) finished with slot 10 (arm
1d83776, seed 3002) crashed by a Metal GPU fault, so one pair is unmatched.
Amendment 5 retains every valid receipt with a result from BOTH earlier runs
(`run-heldout-2026-10-13b` and `run-heldout-2026-10-13c`), verifies their
hashes, and runs only the missing slot in a NEW directory. It reuses
`audit_heldout_resume.py` unchanged (its script hash is frozen in the Amendment
4 plan) and replaces only the single-source retention and the reports, so all
older plans still reproduce. Default mode writes a frozen plan and starts no
training. With --execute-approved and a NEW --output directory it runs only the
missing slots through LocalExecutor. No existing evidence changes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from openloop.ledger import ConflictError, content_hash
from openloop.ledger.validation import plain
from openloop.studies.claims_audit import ClaimedOutcome, Comparison, analyze_audit
from openloop.studies.heldout import (
    analyze_heldout,
    combined_observations,
    heldout_run_observations,
)
from openloop.studies.resume import (
    ResumeMismatchError,
    Retained,
    missing_slots,
    select_retained,
)
from openloop.studies.validity import valid_receipts

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_heldout_resume", Path(__file__).with_name("audit_heldout_resume.py")
)
assert _spec is not None and _spec.loader is not None
resume: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resume)
valid: ModuleType = resume.valid
base: ModuleType = resume.base
audit: ModuleType = resume.audit

STUDY_DIR = REPO / "research/s1-part2"
SOURCE_RUNS = (
    STUDY_DIR / "run-heldout-2026-10-13b",
    STUDY_DIR / "run-heldout-2026-10-13c",
)
AMENDMENT_4_PLAN = STUDY_DIR / "plan-heldout-2026-10-13c.json"
SOURCE_KEY = resume.SOURCE_KEY
ATTEMPTS_REPORT = resume.ATTEMPTS_REPORT
FOUR_PAIR_REPORT = "heldout-report-4pair.json"
STOP_FILES = resume.STOP_FILES
display = resume.display
load_receipt_files = resume.load_receipt_files
why_not_retained = resume.why_not_retained


def preflight(
    source: Path,
    python: Path,
    sources: Sequence[Path] = SOURCE_RUNS,
    *,
    amendment_4_plan: Path = AMENDMENT_4_PLAN,
) -> dict[str, object]:
    """Build the frozen Amendment 5 plan; refuse if any earlier run mismatches."""
    sources = [s.resolve() for s in sources]
    first, last = sources[0], sources[-1]
    # The Amendment 4 plan must reproduce from current sources (which also
    # re-checks the Amendment 3 plan) and match the plan stored by its run.
    prior = json.loads(amendment_4_plan.read_text(encoding="utf-8"))
    fresh_4 = resume.preflight(source, python, first)
    if (
        prior["plan_hash"] != content_hash(prior["plan"])
        or prior["plan_hash"] != fresh_4["plan_hash"]
        or prior["plan"] != plain(fresh_4["plan"])
    ):
        raise ResumeMismatchError(
            "Amendment 4 plan does not reproduce from the current sources"
        )
    run_plan = json.loads((last / "plan.json").read_text(encoding="utf-8"))
    if run_plan != {"plan_hash": prior["plan_hash"], "plan": prior["plan"]}:
        raise ResumeMismatchError("Latest run's plan differs from the Amendment 4 plan")
    plan = dict(fresh_4["plan"])  # type: ignore[call-overload]
    order = plan["order"]
    floors = {
        arm: item["floor"]
        for arm, item in plan["validity"]["reference_steps"].items()  # type: ignore[index]
    }
    retained: list[Retained] = []
    entries: list[dict[str, object]] = []
    not_retained: list[dict[str, object]] = []
    stop_evidence: dict[str, str] = {}
    seen: dict[int, str] = {}
    for directory in sources:
        receipts = load_receipt_files(directory)
        kept = select_retained(receipts, order, floors)  # type: ignore[arg-type]
        for item in kept:
            if item.slot in seen:
                raise ResumeMismatchError(
                    f"Slot {item.slot} has valid receipts in {seen[item.slot]} "
                    f"and {display(directory)}"
                )
            seen[item.slot] = display(directory)
        kept_files = {r.file for r in kept}
        retained.extend(kept)
        entries.extend(
            {**entry, SOURCE_KEY: display(directory)}
            for entry in resume.verify_retained_evidence(directory, kept, floors)
        )
        not_retained.extend(
            {
                "receipt": name,
                SOURCE_KEY: display(directory),
                "receipt_sha256": audit.digest(directory / name),
                "reason": why_not_retained(receipt),
            }
            for name, receipt in receipts.items()
            if name not in kept_files
        )
        stop_evidence.update(
            {
                f"{display(directory)}/{name}": audit.digest(directory / name)
                for name in STOP_FILES
                if (directory / name).is_file()
            }
        )
    if not missing_slots(order, retained):  # type: ignore[arg-type]
        raise ResumeMismatchError("Every slot is already retained; nothing to run")
    entries.sort(key=lambda e: e["slot"])  # type: ignore[arg-type, return-value]
    plan["study"] = (
        "S1 part 2, Amendment 5: completion of the held-out design after a GPU fault"
    )
    plan["resume"] = {
        "source_directories": [display(s) for s in sources],
        "source_plan": {
            "path": display(amendment_4_plan),
            "plan_hash": prior["plan_hash"],
            "file_sha256": audit.digest(amendment_4_plan),
        },
        "shard_sha256": plan["heldout_phase"]["shard"]["sha256"],  # type: ignore[index]
        "rule": "Only receipts with validity.valid true and a successful summary "
        "are retained, and each must match its frozen slot, program hashes and "
        "step floor. Retained slots are never run again.",
        "retained": entries,
        "missing_slots": [
            {k: item[k] for k in ("index", "kind", "arm", "seed")}
            for item in missing_slots(order, retained)  # type: ignore[arg-type]
        ],
        "not_retained": not_retained,
        "stop_evidence": stop_evidence,
    }
    plan["code"] = {
        **plan["code"],  # type: ignore[call-overload]
        "audit_heldout_complete_script_sha256": audit.digest(Path(__file__)),
    }
    return {"plan_hash": content_hash(plan), "plan": plan}


def pending_slots(plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The slots to run: the frozen order minus every retained slot."""
    retained = {e["slot"] for e in plan["resume"]["retained"]}  # type: ignore[index]
    return [item for item in plan["order"] if item["index"] not in retained]  # type: ignore[attr-defined]


def _resolve(path: str) -> Path:
    directory = Path(path)
    return directory if directory.is_absolute() else REPO / directory


def retained_receipts(plan: Mapping[str, object]) -> list[dict[str, object]]:
    """Re-read the retained receipts, check their hashes, and mark their source."""
    result: list[dict[str, object]] = []
    for entry in plan["resume"]["retained"]:  # type: ignore[index]
        path = _resolve(entry[SOURCE_KEY]) / entry["receipt"]
        if audit.digest(path) != entry["receipt_sha256"]:
            raise ResumeMismatchError(f"{entry['receipt']} changed since the plan")
        result.append(
            {
                **json.loads(path.read_text(encoding="utf-8")),
                SOURCE_KEY: entry[SOURCE_KEY],
                "source_receipt": entry["receipt"],
                "source_receipt_sha256": entry["receipt_sha256"],
            }
        )
    return result


def write_reports(output: Path, plan: dict[str, object]) -> None:
    """Write the 5-pair primary reports, the 4-pair reproduction and attempts."""
    retained = retained_receipts(plan)
    here = display(output)
    new = [
        {
            **receipt,
            SOURCE_KEY: here,
            "source_receipt": name,
            "source_receipt_sha256": audit.digest(output / name),
        }
        for name, receipt in load_receipt_files(output).items()
    ]
    receipts = [*retained, *new]
    valid_ones = valid_receipts(receipts)
    earlier = [r for r in valid_ones if r.get(SOURCE_KEY) != here]
    parent = json.loads((base.PARENT_RUN / "plan.json").read_text(encoding="utf-8"))[
        "plan"
    ]
    comparisons = [
        Comparison(
            f"{c['child']} vs {c['parent']}",
            c["child"],
            c["parent"],
            ClaimedOutcome(c["claimed"]),
            c["published_child_bpb"],
            c["published_parent_bpb"],
        )
        for c in parent["comparisons"]
    ]
    completion = [r for r in valid_ones if r.get("kind") == base.KIND_COMPLETION]
    audit.save(
        output / base.SEED_COMPLETION_REPORT,
        analyze_audit(
            comparisons,
            combined_observations(
                list(valid.load_receipts(base.PARENT_RUN)), completion
            ),
        ),
    )
    audit.save(
        output / base.HELDOUT_REPORT,
        {
            "analysis": "primary: all valid attempts from the three directories",
            **plain(
                analyze_heldout(
                    heldout_run_observations(valid_ones),
                    child=base.CHILD,
                    parent=base.PARENT,
                )
            ),
        },
    )
    # The same analysis over the valid attempts that existed before this run;
    # with the Amendment 4 inputs it reproduces that run's heldout-report.json.
    audit.save(
        output / FOUR_PAIR_REPORT,
        {
            "analysis": "secondary: valid attempts retained from the earlier "
            "directories only (the Amendment 4 four-pair result)",
            **plain(
                analyze_heldout(
                    heldout_run_observations(earlier),
                    child=base.CHILD,
                    parent=base.PARENT,
                )
            ),
        },
    )
    scored = [
        r
        for r in valid_ones
        if isinstance(r.get("summary"), Mapping) and "failure" not in r
    ]
    used = {(r[SOURCE_KEY], r["source_receipt"]) for r in scored}
    rows: list[dict[str, object]] = []
    directories = [_resolve(d) for d in plan["resume"]["source_directories"]]  # type: ignore[index]
    for directory in [*directories, output]:
        for name, receipt in load_receipt_files(directory).items():
            rows.append(
                {
                    "slot": receipt.get("index"),
                    "arm": receipt.get("arm"),
                    "seed": receipt.get("seed"),
                    "attempt": receipt.get("attempt"),
                    "source_directory": display(directory),
                    "receipt": name,
                    "receipt_sha256": audit.digest(directory / name),
                    "validity": receipt.get("validity"),
                    "failure": receipt.get("failure"),
                    "has_result": isinstance(receipt.get("summary"), Mapping),
                    "used_in_analysis": (display(directory), name) in used,
                }
            )
    audit.save(
        output / ATTEMPTS_REPORT,
        {
            "attempts": rows,
            "retained": len(retained),
            "used_in_analysis": len(scored),
            "not_used": sum(1 for row in rows if not row["used_in_analysis"]),
            "prior_attempts_not_retained": plan["resume"]["not_retained"],  # type: ignore[index]
        },
    )


@contextlib.contextmanager
def _completion_hooks() -> Iterator[None]:
    """Point the unchanged Amendment 4 executor at the multi-source helpers."""
    names = {
        "pending_slots": pending_slots,
        "retained_receipts": retained_receipts,
        "write_reports": write_reports,
    }
    saved = {n: getattr(resume, n) for n in names}
    try:
        for n, f in names.items():
            setattr(resume, n, f)
        yield
    finally:
        for n, f in saved.items():
            setattr(resume, n, f)


async def execute(
    source: Path,
    python: Path,
    prepared: dict[str, object],
    output: Path,
    *,
    read: audit.PressureReader = audit.read_memory,
) -> None:
    with _completion_hooks():
        await resume.execute(source, python, prepared, output, read=read)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="fork clone")
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="write report files for a finished or interrupted output",
    )
    args = parser.parse_args()
    if args.report:
        plan = json.loads((args.report / "plan.json").read_text())["plan"]
        write_reports(args.report.resolve(), plan)
        return
    if args.source is None or args.worker_python is None:
        parser.error("--source and --worker-python are required")
    source, python = args.source.resolve(), args.worker_python.absolute()
    prepared = preflight(source, python)
    slots = len(pending_slots(prepared["plan"]))  # type: ignore[arg-type]
    if not args.execute_approved:
        if args.plan is None:
            parser.error("preparing requires --plan")
        audit.save(
            args.plan,
            {
                **prepared,
                "prepared_at": datetime.now(UTC).isoformat(),
                "training_runs": slots,
                "native_training_executed": False,
            },
        )
        print(
            f"Prepared {slots} runs, plan hash {prepared['plan_hash']}. "
            "No training executed."
        )
        return
    if args.output is None or args.plan is None:
        parser.error("--execute-approved requires --plan and a new --output directory")
    frozen = json.loads(args.plan.read_text(encoding="utf-8"))
    if frozen["plan_hash"] != prepared["plan_hash"] or frozen["plan"] != plain(
        prepared["plan"]
    ):
        raise ConflictError("Prepared plan or environment changed; prepare again")
    if args.output.resolve() in {s.resolve() for s in SOURCE_RUNS}:
        raise ConflictError("Output must be a new directory, not an earlier run")
    asyncio.run(execute(source, python, prepared, args.output.resolve()))


if __name__ == "__main__":
    main()
