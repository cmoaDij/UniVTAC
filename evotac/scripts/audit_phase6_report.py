"""Audit a completed P6 report against the frozen denominator protocol."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(report, plan, group):
    if report.get("schema") != "evotac.paired_ablation.v1":
        raise ValueError("unsupported paired-ablation report schema")
    if report.get("denominator_scope") == "inferred_from_input_rows":
        raise ValueError("report denominator was inferred from rows; supply the frozen plan")
    methods = tuple(plan["experiments"][group])
    conditions = tuple(plan["condition_codes"].values())
    expected_per_condition = int(plan["parent_scene_count_per_method"])
    seeds = tuple(plan["train_seeds"])
    rows = report.get("rows", [])
    if report.get("plan", {}).get("methods") != list(methods):
        raise ValueError("report method matrix does not match frozen plan")
    counts = Counter((row.get("condition"), row.get("method")) for row in rows)
    parent_sets = defaultdict(set)
    keys = set()
    for row in rows:
        if row.get("condition") not in conditions:
            raise ValueError(f"unknown condition: {row.get('condition')}")
        if row.get("method") not in methods:
            raise ValueError(f"unknown method: {row.get('method')}")
        if int(row.get("seed", -1)) not in seeds:
            raise ValueError("row has a seed outside the frozen train seeds")
        if int(row.get("budget_physics_steps", -1)) != int(plan["budget_physics_steps"]):
            raise ValueError("row changed the paired physics budget")
        key = (row["condition"], row["parent_scene_id"], row["method"], int(row["seed"]))
        if key in keys:
            raise ValueError("duplicate parent/condition/method/seed row")
        keys.add(key)
        parent_sets[(row["condition"], row["method"])].add(row["parent_scene_id"])
        for metric in ("success", "violation"):
            if metric not in row or float(row[metric]) not in (0.0, 1.0):
                raise ValueError(f"row must contain binary {metric}")
    expected_rows = expected_per_condition * len(conditions) * len(methods) * len(seeds)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    for condition in conditions:
        for method in methods:
            key = (condition, method)
            if counts[key] != expected_per_condition * len(seeds):
                raise ValueError(f"wrong row count for {condition}/{method}")
            if len(parent_sets[key]) != expected_per_condition:
                raise ValueError(f"parent scene denominator is not independent for {condition}/{method}")
        method_sets = [parent_sets[(condition, method)] for method in methods]
        if any(parent_set != method_sets[0] for parent_set in method_sets[1:]):
            raise ValueError(f"methods are not paired on the same parents for {condition}")
    return {"status": "valid", "group": group, "rows": len(rows),
            "conditions": list(conditions), "methods": list(methods),
            "parent_count_per_condition": expected_per_condition,
            "train_seeds": list(seeds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--group", required=True,
                        choices=("tactile_recovery", "effect_organization", "continual_adaptation"))
    parser.add_argument("--plan", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/phase6_ablation_plan.json")
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()), json.loads(args.plan.read_text()), args.group)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
