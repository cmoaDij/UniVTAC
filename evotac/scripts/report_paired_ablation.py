"""Validate a completed parent-scene ablation JSONL and emit paired metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evotac.evaluation.paired_ablation import AblationPlan, PairedAblationRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL with one completed episode per method/parent/seed")
    parser.add_argument("--name", default="ablation")
    parser.add_argument("--split", default="test", choices=("train", "dev", "acceptance", "test"))
    parser.add_argument("--budget-physics-steps", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("input JSONL has no episode rows")
    keys = [(str(row["parent_scene_id"]), str(row["method"]), int(row["seed"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("input JSONL contains duplicate parent/method/seed rows")
    methods = tuple(dict.fromkeys(str(row["method"]) for row in rows))
    parents = tuple(dict.fromkeys(str(row["parent_scene_id"]) for row in rows))
    seeds = tuple(dict.fromkeys(int(row["seed"]) for row in rows))
    plan = AblationPlan(args.name, methods, parents, train_seeds=seeds, split=args.split,
                        budget_physics_steps=args.budget_physics_steps)
    indexed = dict(zip(keys, rows))

    def callback(parent, method, seed, budget):
        try:
            return indexed[(parent, method, seed)]
        except KeyError as exc:
            raise ValueError(f"missing paired episode: {parent}/{method}/seed{seed}") from exc

    report = PairedAblationRunner(plan, callback).run()
    report["denominator_scope"] = "inferred_from_input_rows"
    report["denominator_warning"] = ("Methods, parents and seeds were inferred from this file. "
                                      "Use scripts/audit_phase6_report.py with the frozen plan "
                                      "before treating the report as a complete P6 denominator.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
