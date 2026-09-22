"""Guard the published state/action layout and policy handoff behavior."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from evotac.policy.ftp1_adapter import encode_observation, decode_chunk, FTP1Policy


def observation():
    def rgb(value):
        return {"valid": True, "data": np.full((16, 20, 3), value, np.uint8)}
    return {"proprioception": {"joint_position": np.arange(7, dtype=np.float32), "finger_position": [0.01, 0.03]},
            "images": {"camera": {"head": {"rgb": rgb(10)}, "wrist": {"rgb": rgb(20)}},
                       "tactile": {"left_tactile": {"rgb_marker": rgb(30)}, "right_tactile": {"rgb_marker": rgb(40)}}}}


class FTP1AdapterTests(unittest.TestCase):
    def test_input_layout_and_privileged_fields_do_not_affect_model(self):
        obs = observation()
        payload, qpos = encode_observation(obs)
        np.testing.assert_array_equal(payload["state"][0, 9:16], np.arange(7))
        self.assertAlmostEqual(float(payload["state"][0, 44]), 0.01)
        self.assertAlmostEqual(float(qpos[-1]), 0.01)
        unused = np.delete(payload["state"][0], list(range(9, 16))+[44])
        self.assertFalse(unused.any())
        self.assertEqual(payload["right_tactile_gripper"].shape, (1, 2, 224, 224, 3))
        self.assertTrue((payload["right_tactile_gripper"][0, 0] == 30).all())
        self.assertTrue((payload["right_tactile_gripper"][0, 1] == 40).all())
        obs["privileged"] = {"success": True, "object_pose": np.ones(7)}
        again, _ = encode_observation(obs)
        for key in payload:
            np.testing.assert_array_equal(payload[key], again[key])

    def test_invalid_sensor_or_state_is_rejected(self):
        for field in ("image", "nan", "shape"):
            obs = observation()
            if field == "image":
                obs["images"]["tactile"]["left_tactile"]["rgb_marker"]["valid"] = False
            elif field == "nan":
                obs["proprioception"]["joint_position"][0] = np.nan
            else:
                obs["proprioception"]["finger_position"] = [0.01]
            with self.assertRaises(ValueError):
                encode_observation(obs)

    def test_chunk_offsets_use_same_inference_base(self):
        chunk = np.zeros((32, 120), np.float32)
        chunk[0] = 999  # Current-frame prediction must never be executed.
        chunk[1, 9:16], chunk[2, 9:16] = 0.1, 0.2
        chunk[:, 44] = 0.02
        base = np.r_[np.arange(7), 0.01]
        actions = decode_chunk(chunk, base, "mix", count=2)
        np.testing.assert_allclose(actions[0].values[:7], base[:7]+0.1)
        np.testing.assert_allclose(actions[1].values[:7], base[:7]+0.2)
        self.assertAlmostEqual(actions[1].values[7], 0.02)
        self.assertAlmostEqual(decode_chunk(chunk, base, "relative")[0].values[7], 0.03)
        np.testing.assert_allclose(decode_chunk(chunk, base, "absolute")[0].values[:7], 0.1)
        for kwargs in ({"first_index": 0}, {"count": 32}, {"stride": 0}):
            with self.assertRaises(ValueError):
                decode_chunk(chunk, base, "mix", **kwargs)
        chunk[1, 10] = np.nan
        with self.assertRaises(ValueError):
            decode_chunk(chunk, base, "mix")

    def test_handoff_discards_cached_chunk_and_reobserves(self):
        calls = []
        def infer(inputs, prompt):
            calls.append(inputs["state"].copy())
            return np.zeros((32, 120), np.float32), {"call": len(calls)}
        backend = SimpleNamespace(infer=infer, metadata={"action_joint_rep": "mix"})
        policy = FTP1Policy(backend, {"chunk_first_index": 1, "execute_chunk_steps": 2, "chunk_stride": 1})
        obs = observation()
        policy.next_action(obs, "insert")
        changed = deepcopy(obs)
        changed["proprioception"]["joint_position"] += 0.5
        cached = policy.next_action(changed, "insert")
        self.assertEqual(len(calls), 1)
        self.assertEqual(cached.values[0], 0)
        policy.next_action(obs, "insert")
        policy.switch_control()
        fresh = policy.next_action(changed, "insert")
        self.assertEqual(len(calls), 3)
        self.assertEqual(fresh.values[0], 0.5)


if __name__ == "__main__":
    unittest.main()
