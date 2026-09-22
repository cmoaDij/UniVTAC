import copy
import json

import h5py
import numpy as np
import pytest

from evotac.scripts.diagnose_snapshot_policy import (
    action_difference, load_pairs, observation_variants, validate_pair,
)
from evotac.data.rollout_logger import write_tree
from evotac.envs.state_replay import seal_scene


def test_dev_pair_requires_same_parent_and_time():
    meta = {"split": "dev", "parent_scene_id": "parent", "seed": 23,
            "condition": {}, "versions": {"code": "pinned"}}
    obs = {"physics_step": 617, "remaining_physics_steps": 100,
           "remaining_recovery_steps": 120, "sim_time": 617 / 120}
    validate_pair(meta, meta, obs, obs)
    for field, value in (("split", "test"), ("parent_scene_id", "other"), ("versions", {})):
        with pytest.raises(ValueError):
            validate_pair(meta, {**meta, field: value}, obs, obs)
    for field in obs:
        with pytest.raises(ValueError, match="Unaligned"):
            validate_pair(meta, meta, obs, {**obs, field: obs[field] + 1})


def test_modality_interventions_preserve_context_and_do_not_alias():
    source = {"images": {"camera": {"x": np.array([1])}, "tactile": {"x": np.array([2])}},
              "proprioception": {"q": np.array([3])}, "references": {"grasp": np.array([4])}}
    restored = copy.deepcopy(source)
    restored["images"]["camera"]["x"][:] = 10
    restored["images"]["tactile"]["x"][:] = 20
    restored["proprioception"]["q"][:] = 30
    restored["references"]["grasp"][:] = 40
    variants = observation_variants(source, restored)
    assert variants["camera_only"]["images"]["camera"]["x"].item() == 10
    assert variants["camera_only"]["images"]["tactile"]["x"].item() == 2
    assert variants["tactile_only"]["images"]["tactile"]["x"].item() == 20
    assert variants["tactile_only"]["references"]["grasp"].item() == 4
    assert variants["proprioception_only"]["proprioception"]["q"].item() == 30
    variants["source_repeat"]["images"]["camera"]["x"][:] = 99
    assert source["images"]["camera"]["x"].item() == 1
    assert variants["source"]["images"]["camera"]["x"].item() == 1


def test_action_metrics_keep_units_separate_and_reject_invalid_inputs():
    reference = np.zeros((16, 8))
    candidate = reference.copy()
    candidate[:, 2] = 0.1
    candidate[:, 7] = 0.0002
    groups = {"arm_rad": (0, 7), "finger_m": (7, 8)}
    result = action_difference(reference, candidate, groups)
    assert result["arm_rad"]["max_abs"] == 0.1
    assert result["finger_m"]["max_abs"] == 0.0002
    for bad in (np.zeros((15, 8)), np.zeros((16, 7)), np.full((16, 8), np.nan)):
        with pytest.raises(ValueError):
            action_difference(reference, bad, groups)


def test_loader_uses_sealed_boundary_and_rejects_tampering(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    meta = {"split": "dev", "parent_scene_id": "parent", "seed": 23,
            "condition": {}, "versions": {"code": "pinned"}}
    boundary = {"physics_step": 100, "remaining_physics_steps": 100,
                "remaining_recovery_steps": 120, "sim_time": 100 / 120,
                "render": np.array([42], dtype=np.uint8)}
    next_obs = {**boundary, "physics_step": 106, "remaining_physics_steps": 94, "sim_time": 106 / 120}
    for name, branch, observations in (
        ("source", "base", [{**boundary, "render": np.array([0], dtype=np.uint8)}, next_obs]),
        ("restored", "probe_candidate", [boundary, next_obs]),
    ):
        with h5py.File(episodes / f"{name}.h5", "w") as h:
            h.attrs["complete"] = True
            h.attrs["scene_metadata"] = json.dumps({**meta, "branch_id": branch})
            group = h.create_group("observations")
            for index, obs in enumerate(observations):
                write_tree(group.create_group(str(index)), obs)
    scene = seal_scene({"observation": boundary})
    with h5py.File(tmp_path / "source_scene.h5", "w") as h:
        write_tree(h, scene)
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {}}))
    (tmp_path / "checks.json").write_text(json.dumps({
        "source_episode_path": str(episodes / "source.h5"), "prefix_controls": 0,
        "probe_controls": 1, "source_scene_sha256": scene["integrity_sha256"]}))
    _, _, pairs, _ = load_pairs(tmp_path, "probe_candidate")
    assert len(pairs) == 2
    assert pairs[0][0]["render"].item() == 42
    with h5py.File(tmp_path / "source_scene.h5", "r+") as h:
        h["observation"]["render"][...] = np.array([7], dtype=np.uint8)
    with pytest.raises(ValueError, match="integrity"):
        load_pairs(tmp_path, "probe_candidate")
