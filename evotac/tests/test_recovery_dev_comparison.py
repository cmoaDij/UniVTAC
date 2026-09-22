import json

import pytest

from evotac.config import ROOT
from evotac.scripts.compare_recovery_dev import ARMS, schedule, summarize


def test_dev_schedule_is_complete_fixed_and_rejects_test():
    protocol = json.loads((ROOT / "configs/recovery_dev_comparison.json").read_text())
    rows = schedule(protocol)
    assert rows == schedule(protocol)
    assert len(rows) == 40
    assert {(r["seed"], r["arm"]) for r in rows} == {
        (s, a) for s in protocol["seeds"] for a in ARMS}
    with pytest.raises(ValueError):
        schedule({**protocol, "split": "test"})
    with pytest.raises(ValueError):
        schedule({**protocol, "seeds": [0, 0]})


def test_summary_retains_invalid_attempts_and_uses_parent_blocks():
    rows = []
    for arm, outcomes in {"baseline_a": [True, False, True], "baseline_b": [False, False, True],
                          "warmstart": [False, False, False], "trained": [True, True, None]}.items():
        for seed, success in enumerate(outcomes):
            rows.append({"arm": arm, "seed": seed, "status": "finished",
                "valid_trial": success is not None, "success": success is True,
                "reason": "reset_exception" if success is None else "success" if success else "object_lost",
                "recovery_actions": int(arm in {"warmstart", "trained"})})
    report = summarize(rows, 3)
    assert report["arms"]["trained"]["invalid"] == 1
    assert report["arms"]["trained"]["success_rate_valid"] == 1
    assert report["arms"]["trained"]["successes_per_scheduled_attempt"] == 2/3
    comparison = report["comparisons"]["trained_vs_baseline_a"]
    assert comparison["complete_dev_parents"] == 2
    assert comparison["candidate_only_success"] == 1
    assert comparison["reference_only_success"] == 0
    assert comparison["success_rate_difference"] == .5
    assert comparison["exact_mcnemar_two_sided_p_descriptive"] == 1
    assert report["comparisons"]["baseline_b_vs_baseline_a"]["outcome_disagreements"] == 1
    assert report["test_started"] is False
    assert report["strict_snapshot_audit_passed"] is False


def test_empty_summary_does_not_invent_failures_or_p_values():
    report = summarize([], 10)
    assert report["arms"]["trained"]["valid"] == 0
    assert report["arms"]["trained"]["success_rate_valid"] is None
    assert report["comparisons"]["trained_vs_baseline_a"]["exact_mcnemar_two_sided_p_descriptive"] is None
