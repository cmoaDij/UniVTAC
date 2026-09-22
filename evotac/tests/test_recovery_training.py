import numpy as np

from evotac.data.schemas import Action
from evotac.learning.recovery_buffer import ReplayBuffer
from evotac.learning.recovery_rollout import RecoveryCollector
from evotac.learning.recovery_sac import RecoverySAC
from evotac.learning.recovery_training import RecoveryTrainingDriver
from evotac.learning.recovery_trainer import RecoverySACTrainer
from evotac.learning.recovery_observation import recovery_summary, recovery_vector


class Wrapper:
    def __init__(self):
        self.done = False
        self.step_count = 0

    def reset(self, seed, condition=None, **kwargs):
        self.done = False
        self.step_count = 0
        return {"physics_step": 0, "stable": False}, {}

    def step(self, action):
        self.step_count += 1
        if action.kind.value == "recovery_delta":
            stable = self.step_count >= 2
        else:
            stable = False
        terminal = self.step_count >= 3
        self.done = terminal
        obs = {"physics_step": self.step_count * 6, "stable": stable, "id": self.step_count}
        info = {"execution_info": {"valid_trial": True, "status": "executed", "physics_steps": 6,
                                    "bootstrap_allowed": not terminal, "terminated": terminal,
                                    "truncated": False, "reason": "success" if terminal else "running",
                                    "incomplete": False}}
        return obs, float(terminal), terminal, False, info


def test_driver_commits_recovery_fragment_with_continuation_reward():
    wrapper = Wrapper()
    buffer = ReplayBuffer(10)
    collector = RecoveryCollector(wrapper, buffer, lambda obs: np.asarray([obs.get("id", 0), 0], np.float32))

    class Recovery:
        def select_skill(self, observation, summary):
            return "skill"

        def __call__(self, observation, summary, skill):
            return np.zeros(7, np.float32)

    driver = RecoveryTrainingDriver(
        wrapper, collector, feature_summary=lambda obs: {"object_lost_risk": 0.9,
                                                          "remaining_physics_steps": 30,
                                                          "stable": obs.get("stable", False)},
        base_policy=lambda obs: Action("joint_target", [0.0] * 8),
        recovery_policy=Recovery(), max_recovery_actions=4, stable_cycles=1)
    result = driver.run(0)
    assert result.status == "committed"
    assert result.admitted_transitions == 2
    assert len(buffer) == 2
    assert buffer._items[-1].terminated and not buffer._items[-1].bootstrap_allowed


def test_driver_uses_single_skill_fallback_for_plain_callable():
    wrapper = Wrapper()
    buffer = ReplayBuffer(4)
    collector = RecoveryCollector(wrapper, buffer, lambda obs: np.asarray([0.0, 0.0], np.float32))
    driver = RecoveryTrainingDriver(wrapper, collector,
                                    feature_summary=lambda obs: {"object_lost_risk": 0.0,
                                                                  "remaining_physics_steps": 30},
                                    base_policy=lambda obs: Action("joint_target", [0.0] * 8),
                                    recovery_policy=lambda obs, summary, skill: np.zeros(7, np.float32))
    result = driver.run(0)
    assert result.status == "base_terminal"


def test_action_cap_hands_back_and_keeps_true_continuation_reward():
    wrapper = Wrapper()
    buffer = ReplayBuffer(10)
    driver = RecoveryTrainingDriver(
        wrapper, RecoveryCollector(wrapper, buffer, lambda obs: np.zeros(2, np.float32)),
        feature_summary=lambda obs: {"object_lost_risk": 0.9,
                                     "remaining_physics_steps": 30, "stable": False},
        base_policy=lambda obs: Action("joint_target", [0.0] * 8),
        recovery_policy=lambda obs, summary, skill: np.zeros(7, np.float32),
        max_recovery_actions=1, handoff_at_cap=True)
    result = driver.run(0)
    assert result.status == "committed" and result.reason == "success"
    assert result.recovery_actions == result.admitted_transitions == len(buffer) == 1
    assert buffer._items[-1].reward == 0.99
    assert not buffer._items[-1].bootstrap_allowed


def test_trainer_updates_only_admitted_recovery_transitions_and_round_trips(tmp_path):
    wrapper = Wrapper()
    buffer = ReplayBuffer(10, seed=2)
    collector = RecoveryCollector(wrapper, buffer,
                                   lambda obs: np.asarray([obs.get("id", 0), 0], np.float32))
    driver = RecoveryTrainingDriver(
        wrapper, collector, feature_summary=lambda obs: {"object_lost_risk": 0.9,
                                                          "remaining_physics_steps": 30,
                                                          "stable": obs.get("stable", False)},
        base_policy=lambda obs: Action("joint_target", [0.0] * 8),
        recovery_policy=lambda obs, summary, skill: np.zeros(7, np.float32),
        max_recovery_actions=4, stable_cycles=1)
    learner = RecoverySAC(2, 7, hidden_dim=8, seed=3)
    trainer = RecoverySACTrainer(driver, learner, batch_size=2, updates_per_interaction=1)
    result = trainer.run_episode(0)
    assert result.episode.status == "committed"
    assert len(result.updates) == 2
    assert result.skipped_updates == 0
    checkpoint = tmp_path / "trainer.pt"
    trainer.save(checkpoint)
    trainer.load(checkpoint)


def test_recovery_observation_adapter_is_fixed_size_and_observable_only():
    features = {"difference_cues": np.zeros((2, 2, 9), np.float32),
                "proprioception": np.zeros(8, dtype=np.float32),
                "previous_accepted_target": np.arange(9, dtype=np.float32),
                "tactile_age_steps": np.asarray([1, 2], np.float32),
                "remaining_task_steps": 600, "remaining_recovery_steps": 60}
    assert recovery_vector(features).shape == (3162,)
    summary = recovery_summary(features)
    assert summary["remaining_physics_steps"] == 600 and summary["stable"]
