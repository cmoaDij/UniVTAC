from contextlib import contextmanager
import sys

import numpy as np
import pytest

from evotac.config import load_config, validate_config, output_path
from evotac.data.schemas import Action, ActionKind, RobotState
from evotac.envs.action_adapter import ActionAdapter, apply_world_rotvec, rotation_matrix
from evotac.envs.fixed_step_executor import FixedStepExecutor


def state():
    return RobotState(np.array([0, 0, 0, -1, 0, 1, 0.]), np.zeros(7), np.ones(2)*.02,
                      np.zeros(3), np.array([1., 0, 0, 0]), np.array([1., 0, 0, 0]), np.c_[np.eye(6), np.zeros(6)])


def test_pure_imports_and_config():
    assert "isaaclab" not in sys.modules and "envs._base_task" not in sys.modules
    c = load_config()
    for bad in (0, 19, 60, float("nan"), True):
        c["simulation"]["control_hz"] = bad
        with pytest.raises(ValueError):
            validate_config(c)
    for bad in ("../data/oops", "envs/oops", "/tmp/oops"):
        with pytest.raises(ValueError):
            output_path(bad)


def test_actions_reject_bad_contracts():
    for kind, values in (("joint_target", [0]*7), ("recovery_delta", [0]*8), ("oops", [0]*7), ("recovery_delta", [np.nan]*7)):
        with pytest.raises(ValueError):
            Action(kind, values)


def test_ik_frame_equivariance_and_rotvec():
    adapter = ActionAdapter(load_config()["controller"])
    s = state()
    action = Action(ActionKind.RECOVERY_DELTA, [1, 0, 0, 0, 0, 1, 0])
    original = adapter.adapt(action, s)
    s.base_quaternion_world = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    rotated = adapter.adapt(action, s)
    np.testing.assert_allclose(original.joint_target, rotated.joint_target, atol=1e-10)
    np.testing.assert_allclose(rotated.details["base_delta"][:3], [0, -.001, 0], atol=1e-10)
    q = apply_world_rotvec([1, 0, 0, 0], [0, 0, np.pi/2])
    np.testing.assert_allclose(rotation_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-10)


class FakeTask:
    def __init__(self, fail_after=None):
        self._physics_step_count = 0
        self._last_render_physics_step = -1
        self.plan_success = True
        self.render_count = 0
        self.fail_after = fail_after
        self.active = 1

    def read_robot_state(self):
        return state()

    def apply_command(self, command):
        return {"accepted": True, **command.as_dict()}

    @contextmanager
    def _configured_decimation(self):
        self.active = 6
        try:
            yield
        finally:
            self.active = 1

    def _step(self, is_save):
        assert is_save is False
        for _ in range(self.active):
            if self._physics_step_count == self.fail_after:
                raise RuntimeError("sim failure")
            self._physics_step_count += 1
        self._update_render()

    def _update_render(self):
        self.render_count += 1
        self._last_render_physics_step = self._physics_step_count


def test_exact_steps_single_render_and_partial_exception():
    a = Action("recovery_delta", [0]*7)
    adapter = ActionAdapter(load_config()["controller"])
    task = FakeTask()
    result = FixedStepExecutor(task, adapter).execute(a)
    assert result["physics_steps"] == 6 and result["sim_time_end"] == .05
    assert task.render_count == 1 and task.active == 1
    assert result["applied_action"]["force"] is False
    task = FakeTask(fail_after=2)
    result = FixedStepExecutor(task, adapter).execute(a)
    assert result["physics_steps"] == 2 and not result["valid_trial"]
    assert task.active == 1
    task = FakeTask()
    task.plan_success = False
    result = FixedStepExecutor(task, adapter).execute(a)
    assert result["physics_steps"] == 0 and result["status"] == "rejected"


def test_acceptance_compares_controller_precision_without_tolerance():
    from evotac.envs.fixed_step_executor import targets_match_readback
    target = np.array([-2.02345644, 2.25432810])
    actual = target.astype(np.float32)
    assert targets_match_readback({"q": target}, {"q": actual})
    actual[0] = np.nextafter(actual[0], np.float32(0))
    assert not targets_match_readback({"q": target}, {"q": actual})


def test_singular_and_limited_ik_are_rejected_before_physics():
    from evotac.envs.action_adapter import ActionRejected
    c = load_config()["controller"]
    adapter = ActionAdapter(c)
    s = state()
    s.jacobian_world[:] = 0
    with pytest.raises(ActionRejected, match="singularity"):
        adapter.adapt(Action("recovery_delta", [1, 0, 0, 0, 0, 0, 0]), s)
    s = state()
    c["max_joint_delta"] = 1e-7
    with pytest.raises(ActionRejected, match="residual"):
        adapter.adapt(Action("recovery_delta", [1, 0, 0, 0, 0, 0, 0]), s)


def test_joint_and_normalized_clipping_are_auditable():
    adapter = ActionAdapter(load_config()["controller"])
    s = state()
    command = adapter.adapt(Action("joint_target", [100]*8), s)
    np.testing.assert_allclose(command.joint_target-s.joint_position, .04)
    assert command.gripper_target == .039
    assert np.all(command.details["joint_clip_delta"] < 0)
    command = adapter.adapt(Action("recovery_delta", [2, 0, 0, 0, 0, 0, 0]), s)
    assert command.details["normalized_clipped"][0] == 1


def test_zero_step_infrastructure_error_does_not_claim_simulator_acceptance():
    task = FakeTask(fail_after=0)
    result = FixedStepExecutor(task, ActionAdapter(load_config()["controller"])).execute(Action("recovery_delta", [0]*7))
    assert result["physics_steps"] == 0 and not result["valid_trial"]
    assert result["applied_action"]["accepted_by_target_api"] is True
    assert result["applied_action"]["accepted"] is None
    assert result["applied_action"]["simulation_submission_confirmed"] is False
