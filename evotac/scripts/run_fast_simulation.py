"""Run a fast surrogate sweep while Isaac/TacEx GPUs are unavailable.

The output is a control-contract and ablation smoke report.  It deliberately
does not claim physical validity; final EvoTac conclusions still require the
same seeds and conditions through the real Isaac wrapper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evotac.simulation import FastInsertHoleEnv, FastSimulationConfig, run_episode


def _base_policy(env, observation, tactile):
    return ("base", None)


def _recovery_policy(env, observation, tactile):
    features = env.features(observation, tactile=tactile)
    if env.state.recovery_controls < env.config.recovery_horizon_controls and not features["stable"]:
        vision = np.asarray(observation["vision"], dtype=np.float32)
        action = np.zeros(7, dtype=np.float32)
        if features["object_lost_risk"] >= features["contact_blocked"]:
            action[0] = np.clip(vision[0] / 0.35, -1, 1)
            skill = "lateral_adjust"
        else:
            action[1] = np.clip(vision[1] / 0.35, -1, 1)
            skill = "orientation_adjust"
        action[3] = 0.25
        return ("recovery", action, skill)
    return ("base", None)


def _summarize(rows):
    if not rows:
        return {"episodes": 0}
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "object_lost_rate": float(np.mean([row["reason"] == "object_lost" for row in rows])),
        "task_budget_rate": float(np.mean([row["reason"] == "task_budget" for row in rows])),
        "recovery_rate": float(np.mean([row["recovered"] for row in rows])),
        "mean_controls": float(np.mean([row["controls"] for row in rows])),
        "mean_physics_steps": float(np.mean([row["physics_steps"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", default=["nominal", "C1", "C2", "C3"],
                        choices=("nominal", "C1", "C2", "C3"))
    parser.add_argument("--horizon-controls", type=int, default=80)
    parser.add_argument("--recovery-horizon-controls", type=int, default=10)
    parser.add_argument("--no-tactile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(type(seed) is not int or seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be nonnegative integers")
    config = FastSimulationConfig(horizon_controls=args.horizon_controls,
                                  recovery_horizon_controls=args.recovery_horizon_controls)
    tactile = not args.no_tactile
    methods = {"heuristic_base": _base_policy, "heuristic_recovery": _recovery_policy}
    rows = []
    started = time.perf_counter()
    for condition in args.conditions:
        for seed in args.seeds:
            for method, policy in methods.items():
                outcome = run_episode(FastInsertHoleEnv(config), seed, condition, policy, tactile=tactile)
                rows.append({"method": method, "condition": condition, "tactile": tactile, **outcome})
    wall_seconds = time.perf_counter() - started
    summary = {method: _summarize([row for row in rows if row["method"] == method])
               for method in methods}
    report = {
        "schema": "evotac.fast_surrogate.v1",
        "evidence_scope": "control-contract and ablation smoke only; not Isaac/TacEx physics evidence",
        "config": vars(config),
        "seeds": args.seeds,
        "conditions": args.conditions,
        "summary": summary,
        "rows": rows,
        "wall_seconds": wall_seconds,
        "episodes_per_second": len(rows) / wall_seconds if wall_seconds else None,
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
