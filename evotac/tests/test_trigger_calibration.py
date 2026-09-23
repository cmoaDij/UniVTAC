import json

import pytest

from evotac.learning.runtime_monitor import RecoveryMonitor
from evotac.learning.trigger_calibration import fit_trigger_calibration, load_calibration, save_calibration


SHA = "a" * 64


def test_fit_requires_train_and_round_trips(tmp_path):
    records = [{"split": "train", "score": 0.1, "needs_recovery": False},
               {"split": "train", "score": 0.7, "needs_recovery": True},
               {"split": "train", "score": 0.9, "needs_recovery": True}]
    calibration = fit_trigger_calibration(records, source_sha256=SHA, target_recall=1.0)
    assert calibration.split == "train" and calibration.threshold == pytest.approx(0.7)
    path = tmp_path / "calibration.json"
    save_calibration(calibration, path)
    assert load_calibration(path, expected_split="dev") == calibration


def test_fit_rejects_dev_records_and_monitor_exposes_provenance():
    with pytest.raises(ValueError, match="train"):
        fit_trigger_calibration([{"split": "dev", "score": 0.5, "needs_recovery": True}], source_sha256=SHA)
    calibration = fit_trigger_calibration([
        {"score": 0.2, "needs_recovery": False}, {"score": 0.8, "needs_recovery": True}],
        source_sha256=SHA)
    monitor = RecoveryMonitor(calibration=calibration)
    assert monitor.calibrated is True
    assert monitor.provenance()["schema"] == "evotac.trigger_calibration.v1"
    assert monitor.decide({"object_lost_risk": float("nan"), "remaining_physics_steps": 20}).reason == "invalid_risk"
