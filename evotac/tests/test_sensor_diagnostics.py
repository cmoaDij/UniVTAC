from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from evotac.evaluation.sensor_diagnostics import (
    _compare_sensor_diagnostics, _diagnostic_array, _sensor_diagnostics,
)
from evotac.envs.univtac_rl_wrapper import UniVTACRLWrapper


def test_comparison_rejects_different_physics_times_and_missing_fields():
    public = {"physics_step": 12, "uipc_frame": 19}
    source = (public, {"sensor": {"depth": np.array([0.1])}})
    candidate = ({**public, "physics_step": 18, "uipc_frame": 25}, source[1])
    with pytest.raises(ValueError, match="same physics step"):
        _compare_sensor_diagnostics(source, candidate)
    diff = _compare_sensor_diagnostics(source, (public, {}))
    assert not diff["differences"]["sensor.depth"]["compatible"]


def test_sensor_capture_never_triggers_lazy_update_or_aliases_live_data():
    class Sensor:
        _data = NS(output={"rgb": torch.zeros((2, 2, 3), dtype=torch.uint8)})
        _frame = torch.tensor([7])
        _timestamp = torch.tensor([0.5])

        @property
        def data(self):
            raise AssertionError("diagnostics triggered lazy sensor update")

    mesh = np.arange(6).reshape(2, 3).astype(float)
    task = NS(_physics_step_count=12,
              _camera_manager=NS(cameras={"head": Sensor()}),
              _tactile_manager=NS(tactiles={}),
              uipc_sim=NS(world=NS(frame=lambda: 19),
                          sio=NS(simplicial_surface=lambda _: NS(positions=lambda: mesh))))
    diagnostic = _sensor_diagnostics(NS(task=task))
    Sensor._data.output["rgb"].fill_(255)
    mesh.fill(99)
    assert diagnostic[1]["camera.head"]["rgb"].max() == 0
    assert diagnostic[1]["uipc"]["surface_positions"].max() == 5
    assert diagnostic[0]["sensors"]["camera.head"]["frame"] == [7]


def test_nonfinite_buffer_is_not_reported_as_zero_difference():
    summary, raw = _diagnostic_array(np.array([np.nan]))
    assert summary["finite"] is False and raw is None


def test_snapshot_records_rng_before_render_and_current_source_observation():
    rng = np.random.default_rng(42)
    before = deepcopy(rng.bit_generator.state)
    rendered = {}

    def render():
        rendered["marker"] = rng.permutation(63)

    robot = NS(data=NS(**{key: torch.zeros(1, 9) for key in
                         ("joint_pos", "joint_vel", "joint_pos_target", "joint_vel_target")}))
    task = NS(_tactile_manager=NS(tactiles={"right": NS(sensor=NS(
                  marker_motion_simulator=NS(marker_motion_sim=NS(_marker_rng=rng))))}),
              _update_render=render, _robot_manager=NS(robot=robot),
              uipc_sim=NS(save_frame=lambda: None, world=NS(frame=lambda: 19)),
              _actor_manager=NS(actors={}), _physics_step_count=12, phase="base",
              rng=np.random.default_rng(23), references={})
    wrapper = UniVTACRLWrapper.__new__(UniVTACRLWrapper)
    wrapper.task = task
    wrapper.logger, wrapper.done = object(), False
    wrapper.task_elapsed = wrapper.recovery_elapsed = 0
    wrapper.contract = NS(history=[], previous_action=None)
    wrapper.last_observation = {"marker": np.array([-1])}
    wrapper.scene_metadata = {}
    wrapper._observe = lambda: (deepcopy(rendered), {})
    snapshot = wrapper.snapshot_shared_state()
    assert snapshot["marker_rng_states"]["right"] == before
    clone = np.random.default_rng()
    clone.bit_generator.state = snapshot["marker_rng_states"]["right"]
    np.testing.assert_array_equal(clone.permutation(63), snapshot["observation"]["marker"])
    np.testing.assert_array_equal(snapshot["observation"]["marker"], wrapper.last_observation["marker"])
