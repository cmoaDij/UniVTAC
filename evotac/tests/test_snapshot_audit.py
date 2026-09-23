from copy import deepcopy

import numpy as np
import pytest

from evotac.evaluation.snapshot_audit import (continuation_audit, observation_audit,
                                              strict_paired_valid, load_test_gate)


def observation():
    return {"proprioception": {"joint_position": np.zeros(7)},
            "images": {"camera": {"head": {"rgb": {
                "data": np.zeros((2, 2, 3), dtype=np.uint8), "valid": True,
                "sample_physics_step": 6, "age_steps": 0, "format": "RGB_uint8_HWC"}}}},
            "physics_step": 6, "previous_action": {"kind": "joint_target"}}


def test_same_visible_observation_does_not_certify_hidden_state():
    source = observation()
    report = observation_audit(source, deepcopy(source))
    assert report["observations_match"]
    assert not report["valid_match"]


def test_snapshot_identifier_cannot_hide_camera_or_action_drift():
    source = observation()
    candidate = deepcopy(source)
    source["snapshot_digest"] = candidate["snapshot_digest"] = "same"
    candidate["images"]["camera"]["head"]["rgb"]["data"][0, 0, 0] = 10
    candidate["previous_action"] = None
    report = observation_audit(source, candidate)
    assert not report["observations_match"]
    assert not report["metadata_match"]["previous_action"]
    assert report["errors"]["images.camera.head.rgb"]["max_abs"] == 10
    assert not report["valid_match"]


def test_render_range_is_diagnostic_and_cannot_pass_strict_gate():
    observations = [observation() for _ in range(5)]
    observations[4]["images"]["camera"]["head"]["rgb"]["data"][0, 0, 0] = 1
    report = continuation_audit(observations)
    assert report["within_control_image_range"] is False
    assert report["observations_match"] is False
    assert report["valid_match"] is False
    assert not strict_paired_valid({"continuation_probe": report, "branches": []})


def test_test_gate_fails_closed_before_launch(tmp_path):
    report = {"audit_gate": "strict_visible_and_hidden_state.v2", "split": "dev",
              "paired_valid": False, "branches": [], "continuation_probe": {}}
    path = tmp_path / "gate.json"
    path.write_text(__import__("json").dumps(report))
    with pytest.raises(ValueError, match="not valid"):
        load_test_gate(path)
