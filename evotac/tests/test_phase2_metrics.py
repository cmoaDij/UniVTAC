from evotac.scripts.run_phase2_baseline import summarize


def test_baseline_keeps_incomplete_invalid_and_running_denominators():
    rows = [
        {"status": "finished", "controls": 10, "outcome": {"valid_trial": True, "incomplete": False, "reason": "success"}, "task_success": True, "inference_seconds": [.1, .2], "reconstruction": {"valid_match": True}},
        {"status": "finished", "controls": 20, "outcome": {"valid_trial": True, "incomplete": False, "reason": "object_lost"}, "reconstruction": {"valid_match": False}},
        {"status": "finished", "controls": 4, "outcome": {"valid_trial": False, "incomplete": True, "reason": "policy_control_limit"}},
        {"status": "failed"}, {"status": "running"}]
    summary = summarize(rows, 10)
    assert summary["expected_scenes"] == 10 and summary["attempted"] == 5
    assert summary["valid_trials"] == 2 and summary["invalid_or_incomplete"] == 2
    assert summary["running"] == 1
    assert summary["success_rate_valid"] == .5 and summary["object_lost_rate_valid"] == .5
    assert summary["replay_match_rate"] == .5 and summary["inference_seconds"]["n"] == 2


def test_empty_cohort_does_not_report_zero_success_rate():
    summary = summarize([], 10)
    assert summary["success_rate_valid"] is None and summary["replay_match_rate"] is None
