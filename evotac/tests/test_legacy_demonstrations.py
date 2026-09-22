import numpy as np
from evotac.data.legacy_demonstrations import pair_indices, assign_splits, LegacyDemonstrations


def test_three_frame_horizon():
    inputs, targets = pair_indices(11)
    np.testing.assert_array_equal(inputs, [0, 3, 6])
    np.testing.assert_array_equal(targets, [3, 6, 9])
    assert len(pair_indices(3)[0]) == 0


def test_parent_split():
    rows = [{"parent_scene_id": f"parent{i % 10}", "phase": i//10} for i in range(30)]
    assigned = assign_splits(rows)
    assert len({r["split"] for r in assigned if r["parent_scene_id"] == "parent3"}) == 1
    assert {r["parent_scene_id"]: r["split"] for r in assigned} == {r["parent_scene_id"]: r["split"] for r in assign_splits(list(reversed(rows)))}


def test_real_source_seed_and_unknown_provenance():
    legacy = LegacyDemonstrations()
    rows = legacy.manifest()
    row = next(r for r in rows if r["source_episode_id"] == "99")
    assert row["source_seed"] == 761
    assert not row["training_eligible"]
    candidate = legacy.candidate(row)
    assert candidate["target_horizon_seconds"] is None
    assert candidate["future_measured_qpos"].shape[1] == 8
    assert sum(len(legacy.candidate(r)["input_indices"]) for r in rows) == 6315


def test_real_legacy_image_mapping_and_privileged_field_exclusion():
    legacy = LegacyDemonstrations()
    row = legacy.manifest()[0]
    obs = legacy.read_observation(row, 0)
    assert set(obs["camera"]) == {"cam_high", "cam_wrist"}
    assert obs["camera"]["cam_high"]["rgb"].shape == (270, 480, 3)
    assert obs["tactile"]["left_tactile"]["rgb_marker"].shape == (240, 320, 3)
    assert "actor" not in obs and "atom" not in obs
    assert obs["sensor_timestamps"] is None and obs["applied_action"] is None
