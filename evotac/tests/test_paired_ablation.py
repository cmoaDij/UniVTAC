import pytest

from evotac.evaluation.paired_ablation import AblationPlan, PairedAblationRunner, summarize_binary


def test_paired_ablation_keeps_parent_seed_denominator():
    plan = AblationPlan("touch", ("base", "evotac"), ("p0", "p1"), train_seeds=(0, 1), split="test")

    def run(parent, method, seed, budget):
        return {"parent_scene_id": parent, "method": method, "seed": seed, "split": "test",
                "budget_physics_steps": budget, "success": method == "evotac", "violation": False,
                "physics_steps": 6}

    report = PairedAblationRunner(plan, run).run()
    assert len(report["rows"]) == 8
    delta = PairedAblationRunner.paired_delta(report, "evotac", "base")
    assert delta["mean"] == 1.0 and delta["paired_keys"] == 4


def test_paired_ablation_rejects_callback_budget_or_split_changes():
    plan = AblationPlan("touch", ("base",), ("p0",), train_seeds=(0,), split="test")

    def bad(parent, method, seed, budget):
        return {"success": True, "violation": False, "budget_physics_steps": budget + 1}

    with pytest.raises(ValueError, match="budget"):
        PairedAblationRunner(plan, bad).run()


def test_binary_summary_requires_binary_values():
    assert summarize_binary([True, False])["mean"] == 0.5
    with pytest.raises(ValueError):
        summarize_binary([0, 2])


def test_paired_ablation_reports_optional_effect_and_cost_metrics():
    plan = AblationPlan("metrics", ("base",), ("p0", "p1"), train_seeds=(0, 1))

    def run(parent, method, seed, budget):
        return {"success": seed == 1, "violation": False, "physics_steps": 6,
                "wall_seconds": 0.5, "skill_count": 2,
                "effect_success_probability": 0.75, "oracle_success": True}

    summary = PairedAblationRunner(plan, run).run()["summary"]["base"]
    assert summary["effect_brier_score"]["n"] == 4
    assert summary["distance_to_skill_oracle"]["mean"] == 0.5
