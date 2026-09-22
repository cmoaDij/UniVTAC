"""Independently verify handoff from persisted actions, observations and model files."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.envs.state_replay import digest
from evotac.policy.ftp1_adapter import decode_chunk, encode_observation
from evotac.scripts.audit_rollouts import audit_episode


def audit(run_id):
    directory = output_path(Path("runs/phase1")/run_id)
    checks = json.loads((directory/"checks.json").read_text())
    if checks.get("status") != "passed":
        raise ValueError("GPU handoff check did not finish successfully")
    episode = Path(checks["episode_path"])
    general = audit_episode(episode)
    assert general["status"] == "consistent"
    with h5py.File(episode, "r") as handle:
        actions = [read_tree(handle["actions"][k]) for k in sorted(handle["actions"], key=int)]
        observations = [read_tree(handle["observations"][k]) for k in sorted(handle["observations"], key=int)]
        events = [read_tree(handle["events"][k]) for k in sorted(handle["events"], key=int)]
        outcome = json.loads(handle.attrs["outcome"])
    assert len(actions) == 12 and len(observations) == 13
    kinds = [a["proposed_action"]["kind"] for a in actions]
    assert kinds == ["joint_target"]*4 + ["recovery_delta"]*4 + ["joint_target"]*4
    assert all(a["status"] == "executed" and a["valid_trial"] and a["physics_steps"] == 6 for a in actions)
    origin = observations[0]["physics_step"]
    total_budget = observations[0]["remaining_physics_steps"]
    for i, obs in enumerate(observations):
        assert obs["physics_step"] == origin+6*i
        assert obs["remaining_physics_steps"] == total_budget-6*i
        assert sum(obs["history_mask"]) == min(i, 8)
        assert all(m["valid"] and m["age_steps"] == 0 and m["sample_physics_step"] == obs["physics_step"]
                   for sensors in obs["images"].values() for fields in sensors.values() for m in fields.values())
        if i:
            assert digest(obs["previous_action"]) == digest(actions[i-1]["applied_action"])
            assert actions[i-1]["physics_start"] == observations[i-1]["physics_step"]
            assert actions[i-1]["physics_end"] == obs["physics_step"]
    paused = [e for e in events if e["kind"] == "ftp1_paused"]
    resumed = [e for e in events if e["kind"] == "ftp1_resumed"]
    assert len(paused) == len(resumed) == 1
    paused, resumed = paused[0], resumed[0]
    assert paused["cancelled_policy_actions"] == resumed["cancelled_policy_actions"] == 12
    assert paused["physics"] == observations[4]["physics_step"]
    assert resumed["after_physics"] == observations[8]["physics_step"]
    assert resumed["task_elapsed"] == 48 and resumed["recovery_elapsed"] == 24
    assert list(resumed["history_steps"]) == [o["physics_step"] for o in observations[:8]]
    inferences = [json.loads(line) for line in (directory/"policy/inferences.jsonl").read_text().splitlines()]
    assert [i["request_id"] for i in inferences] == [0, 1]
    assert resumed["fresh_inference"]["request_id"] == 1
    max_input_error = max_action_error = 0.0
    for request, start in ((0, 0), (1, 8)):
        payload, base = encode_observation(observations[start])
        with np.load(directory/f"policy/input_{request:05d}.npz") as saved:
            for key, value in payload.items():
                error = float(np.max(np.abs(saved[key].astype(float)-value)))
                max_input_error = max(max_input_error, error)
                assert error == 0, (request, key, error)
        chunk = np.load(directory/f"policy/chunk_{request:05d}.npy")
        decoded = decode_chunk(chunk, base, checks["model"]["action_joint_rep"], count=4)
        for offset, expected in enumerate(decoded):
            actual = actions[start+offset]["proposed_action"]["values"]
            error = float(np.max(np.abs(expected.values-actual)))
            max_action_error = max(max_action_error, error)
            assert error == 0, (request, offset, error)
    assert outcome["episode_cost"]["base"] == 48 and outcome["episode_cost"]["recovery"] == 24
    assert checks["task_elapsed"] == 72 and checks["recovery_elapsed"] == 24
    selected = []
    writer = cv2.VideoWriter(str(directory/"handoff_synchronized.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 240))
    if not writer.isOpened():
        raise RuntimeError("Cannot write handoff evidence video")
    try:
        for i, obs in enumerate(observations):
            panels = [obs["images"]["camera"][name]["rgb"]["data"] for name in ("head", "wrist")]
            panels += [obs["images"]["tactile"][name]["rgb_marker"]["data"] for name in ("left_tactile", "right_tactile")]
            frame = cv2.cvtColor(np.concatenate([cv2.resize(p, (320, 240)) for p in panels], axis=1), cv2.COLOR_RGB2BGR)
            phase = "initial" if i == 0 else "FTP-1 initial chunk" if i <= 4 else "script recovery" if i <= 8 else "FTP-1 fresh chunk"
            cv2.putText(frame, f"obs={i} physics={obs['physics_step']} {phase}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1)
            writer.write(frame)
            if i in (0, 4, 8, 9, 12):
                selected.append(frame)
    finally:
        writer.release()
    if not cv2.imwrite(str(directory/"handoff_contact_sheet.jpg"), np.concatenate(selected)):
        raise RuntimeError("Cannot write handoff contact sheet")
    report = {"status": "passed", "episode_audit": general, "cancelled_actions": 12,
              "total_physics_steps": 72, "base_physics_steps": 48, "recovery_physics_steps": 24,
              "inference_requests": [0, 1], "pause_physics": paused["physics"], "resume_physics": resumed["after_physics"],
              "persisted_input_max_error": max_input_error, "persisted_action_max_error": max_action_error,
              "history_mask_and_previous_action_aligned": True, "sensor_age_steps": 0,
              "first_resumed_chunk_index": 1, "video_fps_simulated_time": 20,
              "scope": "Nonempty real FTP-1 queue handoff; bounded scripted validation, not a recovery success-rate experiment"}
    (directory/"handoff_independent_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_id), indent=2))


if __name__ == "__main__":
    main()
