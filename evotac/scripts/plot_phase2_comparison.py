"""Render completed Phase 2 comparisons without reading live simulation files."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evotac.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    directory = ROOT / "runs/phase2" / args.run_id
    report = json.loads((directory / "comparison.json").read_text())
    if report["status"] != "complete":
        raise ValueError("A completed comparison is required")
    pairs = report["chunk_pairs"]
    colors = {"success": "#168567", "object_lost": "#c74c44", "task_budget": "#d59626"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(pairs))
    for offset, key, label in [(-.2, "baseline", "chunk 1"), (.2, "variant", "chunk 16")]:
        rows = [pair[key] for pair in pairs]
        axes[0].bar(x + offset, [r["controls"] for r in rows], width=.36,
                    color=[colors.get(r["reason"], "gray") for r in rows],
                    hatch="///" if key == "baseline" else None, edgecolor="white", label=label)
    axes[0].set(xticks=x, xticklabels=[p["seed"] for p in pairs], xlabel="Development seed",
                ylabel="Executed controls (20 Hz simulated)", title="Same scene seeds; separate physical resets")
    axes[0].legend()
    for i, key in enumerate(("baseline", "chunk16")):
        bottom = 0
        for reason, color in colors.items():
            count = report[key]["outcome_counts"].get(reason, 0)
            axes[1].bar(i, count, bottom=bottom, color=color, label=reason if i == 0 else None)
            if count:
                axes[1].text(i, bottom + count / 2, str(count), ha="center", va="center", color="white")
            bottom += count
    axes[1].set(xticks=[0, 1], xticklabels=["chunk 1", "chunk 16"], ylabel="Valid trials", title="Outcomes within 200 controls")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=1)
    fig.tight_layout()
    fig.savefig(directory / "paired_outcomes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    grip = report["gripper_seed0_pair"]
    runs = [(grip["baseline"], "chunk 1 / predicted grip"),
            (pairs[0]["variant"], "chunk 16 / predicted grip"),
            (grip["variant"], "chunk 1 / fixed initial grip")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for run, label in runs:
        path = ROOT / "runs/phase1" / run["run_id"] / "diagnosis_timeline.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        times = [(int(row["control"]) + 1) / 20 for row in rows]
        for ax, key in zip(axes, ("ee_z_m", "insertion_rel_z_m", "gripper_target_m")):
            ax.plot(times, [float(row[key]) * 1000 for row in rows], label=f"{label}: {run['reason']}")
    for ax, label in zip(axes, ("End-effector world z (mm)", "Object relative insertion z (mm)", "Single-finger target (mm)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Seed 0: observed trajectories, not identical hidden-state counterfactuals")
    axes[-1].set_xlabel("Simulated seconds after policy control starts")
    fig.tight_layout()
    fig.savefig(directory / "seed0_mechanism.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
