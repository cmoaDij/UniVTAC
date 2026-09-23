"""Audit and plan a tactile-free π0.5 fine-tuning dataset.

The legacy source is deliberately not converted when timing or joint-order
provenance is unconfirmed.  A plan is still written so the missing evidence is
reviewable and a later confirmed conversion can use a new run-id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evotac.config import ROOT
from evotac.data.legacy_demonstrations import LegacyDemonstrations, load_legacy_config


def build_plan(task="insert_hole"):
    config_path = ROOT / "configs/legacy_insert_hole.yaml"
    legacy_config = load_legacy_config(config_path)
    legacy = LegacyDemonstrations(legacy_config)
    rows = [row for row in legacy.manifest()
            if Path(row["source_path"]).parent.parent.name == task]
    if not rows:
        raise ValueError(f"no legacy rows found for task {task!r}")
    split_counts = {split: sum(row["split"] == split for row in rows)
                    for split in ("train", "dev", "test")}
    blockers = []
    if not legacy.config["source_rate_confirmed"]:
        blockers.append("source_rate_confirmed=false")
    if not legacy.config["joint_order_confirmed"]:
        blockers.append("joint_order_confirmed=false")
    if any(row["source_seed"] is None for row in rows):
        blockers.append("missing source seed")
    if len({row["source_seed"] for row in rows}) != len(rows):
        blockers.append("duplicate source seed")
    source_metadata = json.loads((legacy.root / "metadata.json").read_text(encoding="utf-8"))
    results = {result: sum(source_metadata[row["source_episode_id"]].get("result") == result for row in rows)
               for result in sorted({source_metadata[row["source_episode_id"]].get("result") for row in rows})}
    return {
        "schema": "evotac.pi05_dataset_plan.v1",
        "policy": {"name": "pi05", "tactile_inputs": False,
                   "input_modalities": ["camera.head.rgb", "camera.wrist.rgb",
                                         "proprioception.joint_position",
                                         "proprioception.finger_position", "language"],
                   "excluded_modalities": ["tactile.*", "privileged_state.*"]},
        "task": task, "source_root": str(legacy.root),
        "episodes": len(rows), "split_counts": split_counts,
        "source_results": results,
        "source_config": legacy.config,
        "training_eligible": not blockers, "blockers": blockers,
        "action": "blocked_until_source_provenance_is_confirmed" if blockers else "convert_with_new_run_id",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="insert_hole")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if not plan["training_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
