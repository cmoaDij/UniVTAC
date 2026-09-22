from evotac.scripts.validate_phase6_plan import validate
from evotac.scripts.audit_phase6_report import audit


def test_phase6_protocol_is_frozen_and_complete():
    import json
    from pathlib import Path
    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/phase6_ablation_plan.json").read_text())
    result = validate(plan)
    assert result["status"] == "valid"
    assert set(result["groups"]) == {"tactile_recovery", "effect_organization", "continual_adaptation"}
    assert result["parent_scene_count_scope"] == "per_condition"


def test_phase6_report_audit_checks_condition_parent_denominator():
    import json
    from pathlib import Path
    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/phase6_ablation_plan.json").read_text())
    methods = plan["experiments"]["continual_adaptation"]
    rows = []
    for condition in plan["condition_codes"].values():
        for parent in range(2):
            for seed in plan["train_seeds"]:
                for method in methods:
                    rows.append({"condition": condition, "parent_scene_id": f"{condition}_{parent}",
                                 "method": method, "seed": seed,
                                 "budget_physics_steps": plan["budget_physics_steps"],
                                 "success": False, "violation": False})
    small = dict(plan, parent_scene_count_per_method=2)
    report = {"schema": "evotac.paired_ablation.v1",
              "plan": {"methods": methods}, "rows": rows}
    assert audit(report, small, "continual_adaptation")["status"] == "valid"
