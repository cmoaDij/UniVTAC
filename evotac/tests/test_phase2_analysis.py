from copy import deepcopy

import numpy as np
import pytest

from evotac.scripts.analyze_replay_features import pose_discrepancy
from evotac.scripts.compare_phase2_baselines import pair


def test_pose_errors_separate_units_and_ignore_quaternion_sign():
    a = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    b = np.array([.003, .004, 0, -1, 0, 0, 0])
    report = pose_discrepancy(a, b)
    assert report["position_distance_m"] == .005
    assert report["rotation_distance_deg"] == 0
    b[3:] = [np.cos(np.pi/12), 0, 0, np.sin(np.pi/12)]
    assert np.isclose(pose_discrepancy(a, b)["rotation_distance_deg"], 30)
    b[3:] = [1, 0, 0, 1e-5]
    assert pose_discrepancy(a, b)["rotation_distance_deg"] > 0


def test_pair_requires_same_parent_and_physical_and_model_contract():
    row = {"status": "finished", "seed": 0, "parent_scene_id": "seed0", "condition": {"kind": "nominal"},
           "config": {k: {"fixed": 1} for k in ("simulation", "controller", "observation", "budgets", "replay")},
           "policy_config": {k: "fixed" for k in ("source_revision", "checkpoint", "domain", "num_inference_steps", "inference_seed", "prompt")},
           "initial_qpos": [0]*7, "initial_fingers": [.01]*2, "initial_ee": [0]*3}
    variant = deepcopy(row)
    variant["initial_ee"][0] = .001
    assert pair(row, variant)["initial_ee_distance_m"] == .001
    for section, key in (("config", "controller"), ("policy_config", "checkpoint")):
        changed = deepcopy(variant)
        changed[section][key] = "different"
        with pytest.raises(AssertionError):
            pair(row, changed)
    variant["parent_scene_id"] = "seed1"
    with pytest.raises(AssertionError):
        pair(row, variant)
