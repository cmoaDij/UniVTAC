import numpy as np
import pytest

from evotac.simulation import FastInsertHoleEnv, FastSimulationConfig, run_episode


def test_fast_environment_keeps_fixed_step_and_observable_tactile_mask():
    env = FastInsertHoleEnv()
    observation = env.reset(3, "C2")
    assert observation["remaining_physics_steps"] == 480
    assert observation["tactile"].shape == (4,)
    no_tactile = env.observe(tactile=False)
    assert np.all(no_tactile["tactile"] == 0)
    env.step_base(np.zeros(3, dtype=np.float32))
    assert env.state.physics_steps == 6
    assert env.state.controls == 1


def test_fast_recovery_records_skill_and_recovery_budget():
    env = FastInsertHoleEnv(FastSimulationConfig(recovery_horizon_controls=2))
    env.reset(0, "C1")
    env.step_recovery(np.zeros(7, dtype=np.float32), skill_name="lateral_adjust")
    assert env.state.recovered
    assert env.state.selected_skill == "lateral_adjust"
    assert env.state.recovery_controls == 1
    with pytest.raises(ValueError):
        env.step_recovery(np.zeros(6, dtype=np.float32))


def test_fast_episode_callback_keeps_terminal_outcome_and_budget():
    def policy(env, observation, tactile):
        if env.state.condition == "C1" and env.state.recovery_controls == 0:
            return ("recovery", np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32), "lateral_adjust")
        return ("base", None)

    outcome = run_episode(FastInsertHoleEnv(), 4, "C1", policy)
    assert outcome["reason"] in {"success", "object_lost", "task_budget"}
    assert outcome["controls"] == outcome["physics_steps"] // 6
    assert outcome["recovered"]


def test_fast_config_rejects_wrong_decimation():
    with pytest.raises(ValueError, match="decimation"):
        FastSimulationConfig(decimation=3)
