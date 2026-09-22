"""Validate the frozen P6 ablation protocol without starting Isaac."""
import argparse
import json
from pathlib import Path


def validate(plan):
    if plan.get("schema_version") != "evotac.phase6_ablation.v2":
        raise ValueError("unsupported P6 schema")
    if plan.get("split") != "test" or plan.get("train_seeds") != [0, 1, 2]:
        raise ValueError("P6 must use the frozen test split and three seeds")
    count = plan.get("parent_scene_count_per_method", 0)
    if type(count) is not int or count < 100 or plan.get("parent_scene_count_scope") != "per_condition":
        raise ValueError("P6 requires at least 100 independent parent scenes per method")
    experiments = plan.get("experiments", {})
    required = {
        "tactile_recovery": {"vision_act", "tactile_residual_sac", "evotac", "evotac_no_tactile"},
        "effect_organization": {"evotac", "no_metric_loss", "no_effect_retrieval",
                                "no_metric_no_retrieval", "direct_effect_vector_retrieval"},
        "continual_adaptation": {"frozen", "selector_only", "fixed_capacity_finetune", "limited_skill_extension"},
    }
    if set(experiments) != set(required) or any(
            set(experiments[name]) != methods or len(experiments[name]) != len(methods)
            for name, methods in required.items()):
        raise ValueError("P6 method matrix is incomplete or has duplicate methods")
    budget = plan.get("budget_physics_steps")
    if type(budget) is not int or budget <= 0 or budget % 6:
        raise ValueError("P6 budget must be a positive multiple of six physics steps")
    rules = plan.get("rules", {})
    if not all(rules.get(key) is True for key in ("paired_parent_required", "final_test_updates_disabled",
                                                   "infrastructure_failures_separate", "unmatched_replay_excluded_from_effect_labels")):
        raise ValueError("P6 safety and denominator rules are not frozen")
    if rules.get("auto_accept") is not False:
        raise ValueError("P6 cannot automatically accept candidates")
    return {"status": "valid", "schema": plan["schema_version"],
            "groups": {name: len(values) for name, values in experiments.items()},
            "parent_scene_count_per_method": plan["parent_scene_count_per_method"],
            "parent_scene_count_scope": plan["parent_scene_count_scope"],
            "train_seeds": plan["train_seeds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "configs/phase6_ablation_plan.json")
    args = parser.parse_args()
    result = validate(json.loads(args.path.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
