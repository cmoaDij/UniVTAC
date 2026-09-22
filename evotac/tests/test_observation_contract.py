import numpy as np
import pytest

from evotac.config import load_config
from evotac.envs.observation_contract import ObservationContract
from evotac.tests.test_control_contract import state


def test_whitelist_frozen_buffers_staleness_and_future_samples():
    contract = ObservationContract(load_config()["observation"])
    raw = {"observation": {"head": {"rgb": np.zeros((2, 2, 3), np.uint8)}},
           "actor": {"hidden": 123}, "atom": {"tag": "privileged"}}
    stamps = {"camera/head/rgb": 6}
    first = contract.build(raw, state(), 12, stamps, {}, 60)
    raw["actor"]["hidden"] = 999
    second = contract.build(raw, state(), 12, stamps, {}, 60)
    from evotac.envs.state_replay import digest
    assert digest(first) == digest(second)
    assert "actor" not in first and "atom" not in first
    assert first["images"]["camera"]["head"]["rgb"]["valid"]
    raw["observation"]["head"]["rgb"][:] = 100
    assert first["images"]["camera"]["head"]["rgb"]["data"].max() == 0
    old = contract.build(raw, state(), 13, stamps, {}, 60)
    assert not old["images"]["camera"]["head"]["rgb"]["valid"]
    with pytest.raises(ValueError, match="future"):
        contract.build(raw, state(), 5, stamps, {}, 60)


def test_nonfinite_robot_state_is_an_infrastructure_error():
    contract = ObservationContract(load_config()["observation"])
    robot = state()
    robot.ee_position_world[0] = np.nan
    with pytest.raises(ValueError, match="proprioception"):
        contract.build({}, robot, 12, {}, {}, 60)
