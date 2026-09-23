import pytest

from evotac.scripts.prepare_pi05_dataset import build_plan


def test_pi05_plan_rejects_unconfirmed_legacy_provenance():
    plan = build_plan("insert_hole")
    assert plan["policy"]["tactile_inputs"] is False
    assert plan["training_eligible"] is False
    assert "source_rate_confirmed=false" in plan["blockers"]
    assert "joint_order_confirmed=false" in plan["blockers"]
