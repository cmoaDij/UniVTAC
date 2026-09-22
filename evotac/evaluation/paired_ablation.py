"""Run and summarize the minimum EvoTac ablations without changing physics.

The runner deliberately accepts an episode callback instead of importing
Isaac Lab.  The real callback is supplied by a launcher after ``AppLauncher``
starts.  This keeps planning/evaluation import-safe and makes the denominator,
parent-scene split and paired confidence intervals testable on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class AblationPlan:
    name: str
    methods: tuple[str, ...]
    parent_scenes: tuple[str, ...]
    train_seeds: tuple[int, ...] = (0, 1, 2)
    split: str = "test"
    budget_physics_steps: int = 1200

    def __post_init__(self):
        if not self.name or not self.methods or not self.parent_scenes:
            raise ValueError("ablation name, methods and parent scenes are required")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("ablation methods must be unique")
        if len(set(self.parent_scenes)) != len(self.parent_scenes):
            raise ValueError("parent scenes must be unique within a plan")
        if not self.train_seeds or len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("train seeds must be nonempty and unique")
        if self.split not in {"train", "dev", "acceptance", "test"}:
            raise ValueError("unsupported ablation split")
        if type(self.budget_physics_steps) is not int or self.budget_physics_steps <= 0:
            raise ValueError("budget must be a positive integer")


def _normal_ci(values, z=1.96):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    mean = float(values.mean())
    if values.size == 1:
        return {"n": 1, "mean": mean, "lower": None, "upper": None}
    half = float(z * values.std(ddof=1) / np.sqrt(values.size))
    return {"n": int(values.size), "mean": mean, "lower": mean - half, "upper": mean + half}


def summarize_binary(values: Iterable[bool | int | float]):
    """Return a paired-sample mean and 95% normal interval for a binary metric."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size and np.any(~np.isin(array, (0.0, 1.0))):
        raise ValueError("binary metrics must contain only 0/1 values")
    return _normal_ci(array)


def _optional_metric_summary(rows, key):
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"optional metric {key} must be finite")
    return _normal_ci(array)


class PairedAblationRunner:
    """Evaluate each method on the same parent scene and seed combination."""

    def __init__(self, plan: AblationPlan, episode_runner: Callable[[str, str, int, int], Mapping]):
        self.plan = plan
        self.episode_runner = episode_runner

    def run(self):
        rows = []
        for parent_scene in self.plan.parent_scenes:
            for seed in self.plan.train_seeds:
                for method in self.plan.methods:
                    result = dict(self.episode_runner(parent_scene, method, seed, self.plan.budget_physics_steps))
                    if result.get("parent_scene_id", parent_scene) != parent_scene:
                        raise ValueError("episode callback returned a different parent scene")
                    if result.get("method", method) != method:
                        raise ValueError("episode callback returned a different method")
                    if int(result.get("seed", seed)) != seed:
                        raise ValueError("episode callback returned a different seed")
                    if result.get("split", self.plan.split) != self.plan.split:
                        raise ValueError("episode callback returned a different split")
                    if int(result.get("budget_physics_steps", self.plan.budget_physics_steps)) != self.plan.budget_physics_steps:
                        raise ValueError("episode callback changed the paired budget")
                    for key in ("success", "violation"):
                        if key not in result or float(result[key]) not in (0.0, 1.0):
                            raise ValueError(f"episode callback must return binary {key}")
                    rows.append({**result, "parent_scene_id": parent_scene, "method": method,
                                 "seed": seed, "split": self.plan.split,
                                 "budget_physics_steps": self.plan.budget_physics_steps})
        summary = {}
        for method in self.plan.methods:
            subset = [row for row in rows if row["method"] == method]
            summary[method] = {
                "episodes": len(subset),
                "success": summarize_binary(row["success"] for row in subset),
                "violation": summarize_binary(row["violation"] for row in subset),
                "mean_physics_steps": float(np.mean([row.get("physics_steps", 0) for row in subset])),
            }
            for key in ("wall_seconds", "skill_count", "old_condition_retention", "adaptation_curve_auc"):
                metric = _optional_metric_summary(subset, key)
                if metric is not None:
                    summary[method][key] = metric
            predicted = [row for row in subset if "effect_success_probability" in row]
            if predicted:
                if any(float(row["effect_success_probability"]) < 0 or
                       float(row["effect_success_probability"]) > 1 for row in predicted):
                    raise ValueError("effect success probabilities must be in [0, 1]")
                summary[method]["effect_brier_score"] = _normal_ci([
                    (float(row["effect_success_probability"]) - float(row["success"])) ** 2
                    for row in predicted])
            oracle = [row for row in subset if "oracle_success" in row]
            if oracle:
                summary[method]["distance_to_skill_oracle"] = _normal_ci([
                    max(0.0, float(row["oracle_success"]) - float(row["success"]))
                    for row in oracle])
        return {"schema": "evotac.paired_ablation.v1", "plan": self.plan.__dict__,
                "rows": rows, "summary": summary}

    @staticmethod
    def paired_delta(report, candidate: str, baseline: str, metric="success"):
        rows = report["rows"]
        left = {(row["parent_scene_id"], row["seed"]): row for row in rows if row["method"] == candidate}
        right = {(row["parent_scene_id"], row["seed"]): row for row in rows if row["method"] == baseline}
        keys = sorted(set(left) & set(right))
        if not keys:
            return _normal_ci([])
        delta = [float(left[key][metric]) - float(right[key][metric]) for key in keys]
        return {**_normal_ci(delta), "candidate": candidate, "baseline": baseline, "metric": metric,
                "paired_keys": len(keys)}
