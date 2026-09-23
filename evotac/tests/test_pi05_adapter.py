import numpy as np
import pytest

from evotac.policy.pi05_adapter import decode_chunk, encode_observation


def _observation():
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    return {"proprioception": {"joint_position": np.arange(7, dtype=np.float32),
                               "finger_position": np.array([0.01, 0.03], dtype=np.float32)},
            "images": {"camera": {name: {"rgb": {"data": rgb, "valid": True}}
                                    for name in ("head", "wrist")}}}


def test_pi05_input_contract_excludes_tactile():
    inputs, qpos = encode_observation(_observation())
    assert set(inputs) == {"observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb", "observation.state"}
    assert qpos.shape == (8,) and inputs["observation.state"].shape == (8,)


def test_pi05_input_is_invariant_to_tactile_payload():
    observation = _observation()
    first, _ = encode_observation(observation)
    observation["images"]["tactile"] = {"left": {"rgb": {"data": np.full((8, 9, 3), 255, dtype=np.uint8), "valid": True}}}
    second, _ = encode_observation(observation)
    assert set(second) == set(first)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_pi05_relative_action_decode_is_explicit():
    chunk = np.zeros((3, 32), dtype=np.float32)
    chunk[1, :8] = 0.1
    base = np.arange(8, dtype=np.float32)
    action = decode_chunk(chunk, base, representation="relative", first_index=1)[0]
    np.testing.assert_allclose(action.values, base + 0.1)
    with pytest.raises(ValueError, match="absolute or relative"):
        decode_chunk(chunk, base, representation="mix")
