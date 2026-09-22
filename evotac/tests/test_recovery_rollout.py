import numpy as np

from evotac.learning.recovery_buffer import ReplayBuffer
from evotac.learning.recovery_rollout import RecoveryCollector


class FakeWrapper:
    def __init__(self):
        self.done = False
        self.n = 0

    def step(self, action):
        self.n += 1
        obs = {"id": self.n}
        info = {"execution_info": {"valid_trial": True, "status": "executed",
                                    "physics_steps": 6, "bootstrap_allowed": True}}
        return obs, 0.0, False, False, info


def test_handoff_reward_is_added_only_to_final_recovery_transition():
    wrapper = FakeWrapper()
    buffer = ReplayBuffer(10)
    collector = RecoveryCollector(wrapper, buffer, lambda obs: np.asarray([obs["id"], 0], np.float32))
    observation = {"id": 0}
    collector.step(observation, np.zeros(7, np.float32))
    collector.step({"id": 1}, np.ones(7, np.float32) * 0.1)
    assert len(buffer) == 0
    collector.handoff()
    assert collector.finish_continuation({"valid_trial": True, "incomplete": False,
                                           "reason": "success", "terminated": True, "truncated": False}) == 2
    assert len(buffer) == 2
    rows = list(buffer._items)
    assert rows[0].reward == 0.0 and rows[0].bootstrap_allowed
    assert rows[1].reward == 0.99 and rows[1].terminated and not rows[1].bootstrap_allowed
