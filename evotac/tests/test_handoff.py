from types import SimpleNamespace

import numpy as np
import pytest

from evotac.config import load_config
from evotac.data.schemas import Action
from evotac.envs.univtac_rl_wrapper import UniVTACRLWrapper
from evotac.policy.ftp1_adapter import FTP1Policy
from evotac.policy.handoff import scripted_handoff
from evotac.tests.test_rollout_and_replay import WrapperTask, local_dir


def test_next_observation_history_mask_matches_committed_history(local_dir):
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir/"dataset")
    config["logging"]["run_root"] = str(local_dir/"run")
    wrapper = UniVTACRLWrapper(WrapperTask(), config, "history", {"v": 1})
    try:
        obs, _ = wrapper.reset(0)
        assert sum(obs["history_mask"]) == 0
        for count in range(1, 11):
            obs, *_ = wrapper.step(Action("recovery_delta", [0]*7))
            assert sum(obs["history_mask"]) == len(wrapper.contract.history) == min(8, count)
            assert wrapper.contract.history[-1]["observation"]["physics_step"] == obs["physics_step"]-6
    finally:
        wrapper.close()


@pytest.mark.parametrize("terminate", [False, True])
def test_handoff_cancels_real_policy_queue_and_preserves_budgets(local_dir, terminate):
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir/"dataset")
    config["logging"]["run_root"] = str(local_dir/"run")
    task = WrapperTask()
    wrapper = UniVTACRLWrapper(task, config, "handoff", {"v": 1})
    calls = []
    def infer(inputs, prompt):
        calls.append(inputs)
        chunk = np.zeros((32, 120), np.float32)
        chunk[:, 44] = .01
        return chunk, {"request_id": len(calls)-1}
    policy = FTP1Policy(SimpleNamespace(infer=infer, metadata={"action_joint_rep": "mix"}),
                        {"chunk_first_index": 1, "execute_chunk_steps": 16, "chunk_stride": 1})
    try:
        obs, _ = wrapper.reset(0)
        obs, *_ = wrapper.step(policy.next_action(obs, "insert"))
        assert len(policy.pending) == 15
        wrapper.policy_state.action_queue.append(Action("recovery_delta", [1]*7))
        task.failure = terminate
        action, report = scripted_handoff(wrapper, policy, [[0]*7]*2, "insert")
        assert not wrapper.policy_state.action_queue
        if terminate:
            assert action is None and len(calls) == 1 and not policy.pending
            assert wrapper.recovery_elapsed == 6
        else:
            assert len(calls) == 2 and len(policy.pending) == 15
            assert report["cancelled_policy_actions"] == 15
            assert report["after_physics"] == 30
            assert wrapper.task_elapsed == 18 and wrapper.recovery_elapsed == 12
            _, _, _, _, info = wrapper.step(action)
            assert info["execution_info"]["physics_start"] == 30
            assert wrapper.task_elapsed == 24 and wrapper.recovery_elapsed == 12
    finally:
        wrapper.close()
