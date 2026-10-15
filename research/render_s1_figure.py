# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Render the README figure from recorded reports; never open a study ledger.

Run from any directory with `uv run --script research/render_s1_figure.py`.
The reports are read-only inputs. Only docs/figures/s1-summary.{png,svg} is written.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "research/s1/run-2026-10-11/report.json",
    "research/s1-part2/run-heldout-2026-10-13d/heldout-report.json",
)


def main() -> None:
    """Plot the frozen noise estimates and final five-pair audit, with provenance."""
    reports = [json.loads((ROOT / name).read_bytes()) for name in SOURCES]
    noise_report, audit_report = reports
    provenance = "\n".join(
        f"{name}: SHA-256 {hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}"
        for name in SOURCES
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#bdc6d2",
            "axes.labelcolor": "#334155",
            "text.color": "#152238",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "svg.hashsalt": "openloop-s1-summary-v1",
        }
    )
    fig, (noise_ax, audit_ax) = plt.subplots(1, 2, figsize=(12, 6.4))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.29, top=0.73, wspace=0.40)
    fig.suptitle(
        "S1 | Noise floor and an early public-claims audit",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.06,
        0.89,
        "Recorded M4 Pro evidence · October 2026 · BPB = bits per byte",
        color="#475569",
    )

    for kind, label, color, offset in (
        ("fixed_seed", "Fixed seed", "#b45309", 0.12),
        ("distinct_seed", "Distinct seeds", "#0369a1", -0.12),
    ):
        for index, fidelity in enumerate(("8m", "21m")):
            sample = noise_report["noise"][fidelity][kind]
            value = sample["sample_std_bpb"]
            low, high = sample["std_bootstrap_95_percentile_interval"]
            noise_ax.errorbar(
                value,
                index + offset,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
                label=label if index == 0 else None,
            )
            noise_ax.annotate(
                f"{value:.4f}",
                (value, index + offset),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=color,
            )
    noise_ax.set(
        title="A  Legacy AdamW recipe",
        xlabel="Per-run sample SD (BPB)",
        yticks=[0, 1],
        yticklabels=["8m tokens", "21m tokens"],
        xlim=(0, 0.055),
        ylim=(-0.5, 1.7),
    )
    noise_ax.legend(loc="upper left", frameon=False, fontsize=10)
    noise_ax.xaxis.set_major_locator(MultipleLocator(0.01))
    noise_ax.grid(axis="x", color="#e2e8f0")
    noise_ax.set_axisbelow(True)

    heldout = audit_report["heldout"]
    seeds, gains = heldout["seeds"], heldout["gains"]
    audit_ax.scatter(gains, range(len(seeds)), color="#0f766e", s=45, zorder=3)
    mean, (low, high) = heldout["mean_gain"], heldout["interval"]
    audit_ax.errorbar(
        mean,
        -1.4,
        xerr=[[mean - low], [high - mean]],
        fmt="D",
        color="#152238",
        capsize=5,
        markersize=7,
    )
    audit_ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1)
    audit_ax.axhline(-0.7, color="#e2e8f0", linewidth=1)
    audit_ax.set(
        title="B  Published winner vs parent",
        xlabel="Held-out gain (BPB; positive = better)",
        yticks=[-1.4, *range(len(seeds))],
        yticklabels=["Mean / 95% t CI", *map(str, seeds)],
        xlim=(-0.002, 0.033),
        ylim=(-2.2, 4.8),
    )
    audit_ax.xaxis.set_major_locator(MultipleLocator(0.01))
    audit_ax.grid(axis="x", color="#e2e8f0")
    audit_ax.set_axisbelow(True)
    audit_ax.text(
        0.98,
        0.02,
        f"{heldout['wins']}/{len(seeds)} seed wins · sign p = {heldout['sign_p']:.5f}",
        transform=audit_ax.transAxes,
        ha="right",
        fontsize=10,
    )

    rho = noise_report["rank_correlation"]["rho"]
    fig.text(
        0.06,
        0.17,
        "A: 10 executions per point; 70 runs total. "
        f"Screen/full rank rho = {rho:.2f}.\n"
        "Bars: descriptive 95% bootstrap intervals. "
        "This is not current-recipe calibration.",
        fontsize=10,
        linespacing=1.7,
    )
    fig.text(
        0.06,
        0.065,
        "B: 1d83776 vs bc5cdc1; five paired seeds on ClimbMix. "
        f"Mean gain {mean:+.5f} BPB.\n"
        "One lineage; different data and budget from A. "
        "See linked reports for amendments and failures.",
        fontsize=10,
        linespacing=1.7,
    )

    output = ROOT / "docs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    description = (
        "S1 noise estimates and final five-pair held-out audit.\n" + provenance
    )
    fig.savefig(
        output / "s1-summary.png",
        dpi=160,
        metadata={"Description": description, "Software": "Matplotlib 3.10.7"},
    )
    fig.savefig(
        output / "s1-summary.svg", metadata={"Date": None, "Description": description}
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
