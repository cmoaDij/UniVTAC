import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from evotac.config import load_config
from evotac.envs.univtac_rl_wrapper import UniVTACRLWrapper
from evotac.policy.ftp1_adapter import FTP1Policy
from evotac.policy.handoff import scripted_handoff
from evotac.scripts import audit_ftp1_handoff
from evotac.tests.test_rollout_and_replay import WrapperTask, local_dir


def test_handoff_file_audit_detects_wrong_fresh_chunk_and_observation_time(local_dir, monkeypatch):
    directory = local_dir/"evidence"
    (directory/"policy").mkdir(parents=True)
    monkeypatch.setattr(audit_ftp1_handoff, "output_path", lambda path: directory)
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir/"dataset")
    config["logging"]["run_root"] = str(local_dir/"run")
    requests = []
    def infer(inputs, prompt):
        request = len(requests)
        requests.append({"request_id": request})
        chunk = np.zeros((32, 120), np.float32)
        chunk[:, 9:16] = (request+1)*.001
        chunk[:, 44] = .02
        np.savez(directory/f"policy/input_{request:05d}.npz", **inputs)
        np.save(directory/f"policy/chunk_{request:05d}.npy", chunk)
        (directory/"policy/inferences.jsonl").write_text("\n".join(json.dumps(r) for r in requests))
        return chunk, requests[-1]
    policy = FTP1Policy(SimpleNamespace(infer=infer, metadata={"action_joint_rep": "mix"}),
                        {"chunk_first_index": 1, "execute_chunk_steps": 16, "chunk_stride": 1})
    wrapper = UniVTACRLWrapper(WrapperTask(), config, "files", {"v": 1})
    try:
        obs, _ = wrapper.reset(0)
        for _ in range(4):
            obs, *_ = wrapper.step(policy.next_action(obs, "insert"))
        action, _ = scripted_handoff(wrapper, policy, [[0]*7]*4, "insert")
        obs, *_ = wrapper.step(action)
        for _ in range(3):
            obs, *_ = wrapper.step(policy.next_action(obs, "insert"))
        wrapper.stop("handoff_validation_complete")
        checks = {"status": "passed", "episode_path": str(wrapper.logger.path),
                  "task_elapsed": wrapper.task_elapsed, "recovery_elapsed": wrapper.recovery_elapsed,
                  "model": {"action_joint_rep": "mix"}}
        (directory/"checks.json").write_text(json.dumps(checks))
    finally:
        wrapper.close()
    assert audit_ftp1_handoff.audit("files")["status"] == "passed"
    chunk_path = directory/"policy/chunk_00001.npy"
    fresh = np.load(chunk_path)
    stale = np.load(directory/"policy/chunk_00000.npy")
    np.save(chunk_path, stale)
    with pytest.raises(AssertionError):
        audit_ftp1_handoff.audit("files")
    np.save(chunk_path, fresh)
    with h5py.File(checks["episode_path"], "r+") as handle:
        handle["observations/8/physics_step"][()] += 6
    with pytest.raises(AssertionError):
        audit_ftp1_handoff.audit("files")
