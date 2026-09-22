import numpy as np

from evotac.learning.recovery_buffer import ReplayBuffer, Transition
from evotac.learning.recovery_sac import MultiSkillRecoverySAC, RecoverySAC
from evotac.learning.runtime_monitor import RecoveryMonitor
from evotac.learning.skill_registry import initial_recovery_registry, planned_recovery_registry


def test_buffer_round_trip_and_capacity(tmp_path):
    buffer = ReplayBuffer(2, seed=3)
    for index in range(3):
        obs = np.full(4, index, dtype=np.float32)
        buffer.add(Transition(obs, np.zeros(2), index, obs + 1, False))
    assert len(buffer) == 2
    path = tmp_path / "buffer.npz"
    buffer.save(path)
    restored = ReplayBuffer.load(path)
    assert len(restored) == 2
    assert restored.sample(1)["observation"].shape == (1, 4)
    assert bool(restored.sample(1)["bootstrap_allowed"][0]) in {True, False}


def test_sac_action_and_one_update():
    rng = np.random.default_rng(1)
    buffer = ReplayBuffer(8)
    for _ in range(8):
        obs = rng.normal(size=5).astype(np.float32)
        buffer.add(Transition(obs, np.tanh(rng.normal(size=3)), 0.1, rng.normal(size=5), False))
    learner = RecoverySAC(5, 3, hidden_dim=16, seed=1)
    assert learner.act(np.zeros(5, dtype=np.float32)).shape == (3,)
    update = learner.update(buffer.sample(8))
    assert update.batch_size == 8 and np.isfinite(update.critic_loss)


def test_initial_registry_and_monitor_are_explicit():
    registry = initial_recovery_registry()
    assert registry.names() == ("small_lift_adjust_reapproach",)
    assert planned_recovery_registry().names() == (
        "small_lift_adjust_reapproach", "lateral_align", "orientation_adjust")
    assert registry.get(registry.names()[0]).action_dim == 7
    assert RecoveryMonitor().decide({"remaining_physics_steps": 6}).reason == "insufficient_budget"


def test_multi_skill_sac_keeps_independent_learners_and_checkpoint():
    learner = MultiSkillRecoverySAC(("lift", "align", "orient"), 5, hidden_dim=8, seed=4)
    assert learner.act("lift", np.zeros(5, np.float32)).shape == (7,)
    assert learner.learners["lift"] is not learner.learners["align"]
    state = learner.state_dict()
    restored = MultiSkillRecoverySAC(("lift", "align", "orient"), 5, hidden_dim=8, seed=4)
    restored.load_state_dict(state)
    np.testing.assert_allclose(learner.act("orient", np.zeros(5, np.float32), deterministic=True),
                               restored.act("orient", np.zeros(5, np.float32), deterministic=True))
