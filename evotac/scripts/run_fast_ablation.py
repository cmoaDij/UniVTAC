"""Run the frozen P6 method matrix against the fast surrogate.

This command checks parent/condition/seed denominators, method pairing,
budgets and JSON reporting while Isaac/TacEx GPUs are unavailable.  It is a
protocol smoke only: its method callbacks are deterministic heuristics and do
not constitute evidence for learned EvoTac effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from evotac.evaluation.paired_ablation import AblationPlan, PairedAblationRunner
from evotac.simulation import FastInsertHoleEnv, FastSimulationConfig, run_episode
from evotac.scripts.run_fast_simulation import _base_policy, _recovery_policy


CONDITIONS = ("nominal", "C1", "C2", "C3")


def _recovery_variant(method: str, *, tactile: bool):
    """Create an explicitly named heuristic callback for protocol smoke."""
    if method == "evotac":
        return _recovery_policy

    def callback(env, observation, available_tactile):
        features = env.features(observation, tactile=tactile and available_tactile)
        if env.state.recovery_controls >= env.config.recovery_horizon_controls or features["stable"]:
            return ("base", None)
        vision = np.asarray(observation["vision"], dtype=np.float32)
        action = np.zeros(7, dtype=np.float32)
        if method in {"no_effect_retrieval", "direct_effect_vector_retrieval"}:
            # A deterministic fixed ordering stands in for a retrieval policy;
            # real learned retrieval is supplied by the Isaac launcher later.
            action[0 if method == "no_effect_retrieval" else 1] = np.clip(
                vision[0 if method == "no_effect_retrieval" else 1] / 0.35, -1, 1)
        elif method in {"no_metric_loss", "fixed_capacity_finetune"}:
            action[0] = np.clip(vision[0] / 0.35, -1, 1)
        elif method in {"no_metric_no_retrieval", "limited_skill_extension"}:
            action[1] = np.clip(vision[1] / 0.35, -1, 1)
        else:
            raise ValueError(f"unknown surrogate recovery method: {method}")
        action[3] = 0.25
        return ("recovery", action, method)

    return callback


def _method_policy(group: str, method: str):
    if group == "tactile_recovery":
        if method == "vision_act":
            return _base_policy, False
        if method == "evotac_no_tactile":
            return _recovery_policy, False
        return (_recovery_policy if method in {"evotac", "tactile_residual_sac"}
                else _base_policy), True
    if group == "effect_organization":
        return _recovery_variant(method, tactile=True), True
    if group == "continual_adaptation":
        if method == "frozen":
            return _base_policy, True
        return _recovery_variant(method, tactile=True) if method != "selector_only" else _recovery_policy, True
    raise ValueError(f"unknown P6 group: {group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("tactile_recovery", "effect_organization",
                                             "continual_adaptation"), default="tactile_recovery")
    parser.add_argument("--parent-count-per-condition", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if type(args.parent_count_per_condition) is not int or args.parent_count_per_condition < 1:
        raise ValueError("parent-count-per-condition must be positive")
    if any(type(seed) is not int or seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be nonnegative integers")
    plan_data = json.loads((Path(__file__).resolve().parents[1] / "configs/phase6_ablation_plan.json").read_text())
    methods = tuple(plan_data["experiments"][args.group])
    parents = tuple(
        f"fast_parent_{condition}_{index:04d}"
        for condition in CONDITIONS
        for index in range(args.parent_count_per_condition)
    )
    budget = int(plan_data["budget_physics_steps"])
    plan = AblationPlan(f"fast_{args.group}", methods, parents,
                        train_seeds=tuple(args.seeds), split="test",
                        budget_physics_steps=budget)
    config = FastSimulationConfig(horizon_controls=min(80, max(1, budget // 6)))
    started = time.perf_counter()

    def episode_runner(parent_scene, method, seed, paired_budget):
        condition = parent_scene.split("_")[2]
        policy, tactile = _method_policy(args.group, method)
        # Parent identity is part of the reset protocol.  Derive a stable
        # surrogate reset seed from both parent and training seed; reusing only
        # ``seed`` would silently run the same parent 100 times.
        digest = hashlib.sha256(f"{parent_scene}|{seed}".encode()).digest()
        parent_seed = int.from_bytes(digest[:8], "little") % (2**32)
        outcome = run_episode(FastInsertHoleEnv(config), parent_seed, condition, policy, tactile=tactile)
        return {
            "parent_scene_id": parent_scene,
            "method": method,
            "seed": seed,
            "parent_seed": parent_seed,
            "split": "test",
            "budget_physics_steps": paired_budget,
            "success": bool(outcome["success"]),
            "violation": bool(outcome["reason"] == "object_lost"),
            "physics_steps": outcome["physics_steps"],
            "condition": condition,
            "recovered": outcome["recovered"],
            "evidence_scope": "surrogate_protocol_smoke",
        }

    report = PairedAblationRunner(plan, episode_runner).run()
    report.update({
        "evidence_scope": "fast surrogate protocol smoke only; not Isaac/TacEx physical evidence",
        "parent_count_per_condition": args.parent_count_per_condition,
        "conditions": list(CONDITIONS),
        "wall_seconds": time.perf_counter() - started,
    })
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
