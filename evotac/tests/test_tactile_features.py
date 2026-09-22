import numpy as np
import pytest

from evotac.envs.state_replay import digest
from evotac.policy.tactile_features import FrozenTactileEncoder, recovery_features, SENSORS


class EncoderProbe:
    version = {"test": True}
    def __call__(self, images):
        assert images.shape == (6, 4, 5, 3)
        assert images.dtype == np.uint8
        return np.repeat(images.mean((1, 2, 3))[:, None], 512, axis=1)


def sample():
    image = np.ones((4, 5, 3), np.uint8)
    return {"images": {"tactile": {s: {"rgb_marker": {"data": image*20, "valid": True,
                                                       "sample_physics_step": 18, "age_steps": 0}} for s in SENSORS}},
            "references": {k: {"valid": True, "physics_step": step,
                                "images": {s: {"rgb_marker": image*value} for s in SENSORS}}
                           for k, step, value in [("empty", 6, 30), ("grasp", 12, 10)]},
            "physics_step": 18, "remaining_physics_steps": 100,
            "proprioception": {k: np.zeros(n) for k, n in [("joint_position", 7), ("joint_velocity", 7),
                                   ("finger_position", 2), ("ee_position_world", 3), ("ee_quaternion_world", 4)]},
            "previous_action": None, "history_mask": [False]*8}


def test_recovery_features_preserve_signed_cues_reference_order_and_isolation():
    obs = sample()
    result = recovery_features(obs, EncoderProbe(), recovery_remaining_steps=24)
    assert result["tactile_embeddings"].shape == (3, 2, 512)
    np.testing.assert_allclose(result["tactile_embeddings"][:, 0, 0], [20, 30, 10])
    assert (result["difference_cues"][0, :, :3] < 0).all()
    assert (result["difference_cues"][1, :, :3] > 0).all()
    obs.update(actor={"truth": np.ones(7)}, task_success=True, condition="failure")
    assert digest(result) == digest(recovery_features(obs, EncoderProbe(), recovery_remaining_steps=24))
    assert result["remaining_recovery_steps"] == 24 and not result["previous_action_valid"]


@pytest.mark.parametrize("case", ["missing", "future", "reference_order", "negative_budget"])
def test_invalid_reference_and_budget_rejected(case):
    obs = sample()
    if case == "missing":
        obs["references"]["empty"] = None
    elif case == "future":
        obs["images"]["tactile"][SENSORS[0]]["rgb_marker"]["sample_physics_step"] = 19
    elif case == "reference_order":
        obs["references"]["empty"]["physics_step"] = 12
    with pytest.raises(ValueError):
        recovery_features(obs, EncoderProbe(), recovery_remaining_steps=-1 if case == "negative_budget" else 24)


def test_checkpoint_absent_or_wrong_digest_never_falls_back_to_random(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrozenTactileEncoder(tmp_path/"missing.pth", "unknown")
    checkpoint = tmp_path/"bad.pth"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        FrozenTactileEncoder(checkpoint, "wrong")
