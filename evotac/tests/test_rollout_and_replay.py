import json
import shutil
import uuid

import h5py
import numpy as np
import pytest

from evotac.config import ROOT, load_config
from evotac.data.rollout_logger import RolloutLogger, read_tree
from evotac.data.schemas import Action
from evotac.envs.state_replay import PolicyState, seal_scene, validate_scene, calibrate_tolerances
from evotac.envs.univtac_rl_wrapper import UniVTACRLWrapper
from evotac.tests.test_control_contract import FakeTask, state


@pytest.fixture
def local_dir():
    path = ROOT / ".cache" / ("test_"+uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


def test_reset_exception_and_incomplete_file(local_dir):
    path = local_dir / "failure.h5"
    logger = RolloutLogger(path, {"seed": 0})
    logger.event("reset_exception", {"error": "test"})
    logger.close()
    with h5py.File(path) as h:
        assert not h.attrs["complete"]
        assert read_tree(h["events/0"])["kind"] == "reset_exception"
        assert len(h["observations"]) == 0


def test_logger_pairing_and_zero_step(local_dir):
    logger = RolloutLogger(local_dir / "pairs.h5", {"seed": 0})
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    logger.initial_observation({"image": image}, {})
    image[:] = 22
    logger.append({"physics_steps": 6}, {"image": image}, {"terminated": False}, {})
    logger.append({"physics_steps": 0}, {"image": image}, {"terminated": True}, {})
    logger.finish({"reason": "execution_rejected"})
    logger.close()
    with h5py.File(local_dir / "pairs.h5") as h:
        assert h.attrs["complete"]
        assert len(h["observations"]) == 3 and len(h["actions"]) == 2
        assert h["observations/0/image"][:].max() == 0
        assert h["observations/1/image"][:].max() == 22
        assert not h["transitions/1/is_executed_transition"][()]
        assert h["transitions/1/next_observation_index"][()] == 2


def test_scene_tamper_version_split_and_cache():
    versions = {"controller": "v1"}
    scene = seal_scene({"versions": versions, "prefix": [], "parent_scene_id": "p", "split": "dev"})
    validate_scene(scene, versions, {})
    with pytest.raises(ValueError, match="split"):
        validate_scene(scene, versions, {"p": "train"})
    with pytest.raises(ValueError, match="version"):
        validate_scene(scene, {"controller": "v2"}, {})
    scene["prefix"].append({"kind": "hold"})
    with pytest.raises(ValueError, match="integrity"):
        validate_scene(scene, versions, {})
    policy = PolicyState()
    policy.action_queue.extend([1, 2, 3])
    policy.history.append({"observation": 1})
    exported = policy.export()
    policy.switch_control()
    assert not policy.action_queue
    policy.import_state(exported)
    assert list(policy.history) == [{"observation": 1}] and not policy.action_queue


class WrapperTask(FakeTask):
    def __init__(self):
        super().__init__()
        self.ledger = {"initialization": 0, "prefix": 0, "stabilization": 0, "base": 0, "recovery": 0}
        self.metadata = {}
        self.sample_steps = {}
        self.references = {}
        self.prefix = []
        self.phase = "base"
        self.success = self.failure = self.reset_fail = False
        self.closed = False

    def reset(self, seed):
        if self.reset_fail:
            raise RuntimeError("reset test exception")
        self._physics_step_count = 12
        self.ledger["initialization"] += 12
        self.initial_prefix_state = state().as_dict()
        self.initial_prefix_physics = 12
        self._update_render()

    def _step(self, is_save):
        start = self._physics_step_count
        try:
            super()._step(is_save)
        finally:
            self.ledger[self.phase] += self._physics_step_count-start

    def _update_render(self):
        super()._update_render()
        for category, names, fields in (("camera", ["head", "wrist"], ["rgb"]), ("tactile", ["left_tactile", "right_tactile"], ["rgb", "rgb_marker"])):
            for name in names:
                for field in fields:
                    self.sample_steps[f"{category}/{name}/{field}"] = self._physics_step_count

    def _get_observations(self):
        image = np.ones((2, 2, 3), np.uint8)
        return {"observation": {name: {"rgb": image} for name in ("head", "wrist")},
                "tactile": {name: {"rgb": image, "rgb_marker": image} for name in ("left_tactile", "right_tactile")},
                "actor": {"prism": np.zeros(7)}}

    def check_success(self):
        return self.success

    def check_early_stop(self):
        return self.failure

    def close(self):
        self.closed = True


@pytest.mark.parametrize("outcome", ["success", "object_lost", "task_budget", "execution_rejected", "infrastructure_exception", "reset_exception"])
def test_wrapper_all_outcomes(local_dir, outcome):
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir / "dataset")
    config["logging"]["run_root"] = str(local_dir / "run")
    config["budgets"]["task_physics_steps"] = 6
    task = WrapperTask()
    wrapper = UniVTACRLWrapper(task, config, "test", {"v": 1})
    if outcome == "reset_exception":
        task.reset_fail = True
        with pytest.raises(RuntimeError):
            wrapper.reset(0)
    else:
        wrapper.reset(0)
        task.success = outcome == "success"
        task.failure = outcome == "object_lost"
        task.plan_success = outcome != "execution_rejected"
        if outcome == "infrastructure_exception":
            task.fail_after = 14
        _, reward, terminated, truncated, info = wrapper.step(Action("recovery_delta", [0]*7))
        assert info["execution_info"]["reason"] == outcome
        assert reward == float(outcome == "success")
        assert not info["execution_info"]["bootstrap_allowed"]
        if outcome == "execution_rejected":
            assert info["execution_info"]["physics_steps"] == 0
        if outcome == "infrastructure_exception":
            assert info["execution_info"]["physics_steps"] == 2
            assert not info["execution_info"]["valid_trial"]
        with pytest.raises(RuntimeError, match="ended"):
            wrapper.step(Action("recovery_delta", [0]*7))
    file = wrapper.logger.path
    wrapper.close()
    assert task.closed
    with h5py.File(file) as handle:
        assert json.loads(handle.attrs["outcome"])["reason"] == outcome
        assert handle.attrs["complete"]


def test_calibration_requires_independent_repeats():
    with pytest.raises(ValueError):
        calibrate_tolerances([], {"p"}, "v1")


def test_hdf5_scene_round_trip_preserves_integrity(local_dir):
    from evotac.data.rollout_logger import write_tree
    scene = seal_scene({"versions": {"v": 1}, "prefix": [
        {"kind": "hold", "physics_steps": 1, "targets": {"force": False, "arm_position": np.ones(7), "arm_velocity": np.zeros(7), "finger_position": np.zeros(2), "finger_velocity": np.zeros(2)}}],
        "parent_scene_id": "p", "split": "dev", "history": [{"image": np.ones((4, 5, 3), np.uint8), "missing": None}]})
    with h5py.File(local_dir / "scene.h5", "x") as handle:
        write_tree(handle, scene)
    with h5py.File(local_dir / "scene.h5") as handle:
        restored = read_tree(handle)
        validate_scene(restored, {"v": 1}, {})


def test_same_code_episode_tree_match_detects_observation_drift(local_dir):
    from evotac.scripts.replay_chunk16_prefix import _episode_tree_match
    from evotac.data.rollout_logger import write_tree

    def write_episode(path, value):
        with h5py.File(path, "x") as handle:
            for name in ("observations", "actions", "training_info", "events"):
                group = handle.create_group(name)
                record = {"value": value, "label": "stable"}
                if name == "actions":
                    record.update(wall_seconds=1.0, wall_control_hz=2.0)
                if name == "events":
                    record["ledger"] = {"prefix": 10}
                write_tree(group.create_group("0"), record)

    first, second = local_dir / "first.h5", local_dir / "second.h5"
    write_episode(first, np.arange(4, dtype=np.float32))
    write_episode(second, np.arange(4, dtype=np.float32))
    assert _episode_tree_match(first, second)["exact_match"]
    with h5py.File(second, "r+") as handle:
        handle["observations/0/value"][0] = 99
    result = _episode_tree_match(first, second)
    assert not result["exact_match"]
    assert result["groups"]["observations"]["mismatched_keys"] == ["0"]


def test_wrapper_replay_preserves_original_action_and_budget(local_dir):
    config = load_config()
    config["replay"].update(tolerance_version="uncalibrated", tolerance_file=None)
    config["logging"]["dataset_root"] = str(local_dir / "dataset")
    config["logging"]["run_root"] = str(local_dir / "run")
    task = WrapperTask()
    wrapper = UniVTACRLWrapper(task, config, "test", {"v": 1})
    wrapper.reset(0)
    image = np.zeros((2, 2, 3), np.uint8)
    task.references = {kind: {"kind": kind, "images": {name: {"rgb": image, "rgb_marker": image} for name in ("left_tactile", "right_tactile")}, "physics_step": 12, "valid": True} for kind in ("empty", "grasp")}
    for _ in range(8):
        wrapper.step(Action("recovery_delta", [0]*7))
    scene = wrapper.export_scene()
    obs, report = wrapper.replay_start(scene)
    assert report["history_match"] and report["time_match"]
    assert not report["valid_match"] and report["status"] == "uncalibrated"
    assert wrapper.recovery_elapsed == 48 and wrapper.task_elapsed == 48
    assert task.ledger["recovery"] == 96
    assert not wrapper.policy_state.action_queue
    assert all(x["action"]["kind"] == "recovery_delta" for x in wrapper.contract.history)
    wrapper.close()


def test_calibration_cannot_declare_an_unobserved_parent():
    report = {"parent_scene_id": "p1", "history_match": True, "time_match": True,
              "valid_sensors": True, "references_match": True,
              "errors": {"q": {"compatible": True, "max_abs": 0.01, "rmse": 0.01}}}
    with pytest.raises(ValueError, match="repetitions"):
        calibrate_tolerances([report]*4, {"p1", "p2"}, "v1")
    reports = [report]*2 + [{**report, "parent_scene_id": "p2"}]*2
    calibrated = calibrate_tolerances(reports, {"p1", "p2"}, "v1")
    assert calibrated["max_abs"]["q"] == .012
    with pytest.raises(ValueError, match="margin"):
        calibrate_tolerances(reports, {"p1", "p2"}, "v1", margin=float("nan"))


def test_diagnostic_code_drift_preserves_actual_provenance(local_dir, monkeypatch):
    config = load_config()
    config["replay"].update(tolerance_version="uncalibrated", tolerance_file=None)
    config["logging"]["dataset_root"] = str(local_dir / "dataset")
    config["logging"]["run_root"] = str(local_dir / "run")
    versions = {"evotac_code_sha256": "current", "action": "v1"}
    wrapper = UniVTACRLWrapper(WrapperTask(), config, "test", versions)
    try:
        wrapper.reset(0)
        scene = wrapper.export_scene()
        scene.pop("integrity_sha256")
        scene["versions"] = {**versions, "evotac_code_sha256": "source"}
        scene = seal_scene(scene)
        with pytest.raises(ValueError, match="version"):
            wrapper.replay_start(scene)
        # Even a perfect numerical match cannot certify a different revision.
        monkeypatch.setattr("evotac.envs.univtac_rl_wrapper.reconstruction_report",
                            lambda *a, **kw: {"valid_match": True, "status": "calibrated"})
        _, report = wrapper.replay_start(scene, diagnostic_code_drift=True)
        assert report["observable_match"] and not report["valid_match"]
        assert not report["versions_match"] and report["status"] == "diagnostic_code_drift"
        assert report["source_versions"]["evotac_code_sha256"] == "source"
        assert report["execution_versions"] == versions
        assert wrapper.scene_metadata["versions"] == versions
        assert json.loads(wrapper.logger.handle.attrs["scene_metadata"])["versions"] == versions
        tampered = {k: v for k, v in scene.items() if k != "integrity_sha256"}
        tampered["versions"] = {**scene["versions"], "action": "v2"}
        with pytest.raises(ValueError, match="version"):
            wrapper.replay_start(seal_scene(tampered), diagnostic_code_drift=True)
    finally:
        wrapper.close()


def test_diagnostic_implementation_drift_is_explicit(local_dir):
    versions = {"config_sha256": "current", "evotac_code_sha256": "current", "upstream_files_sha256": "current", "action": "v1"}
    scene = seal_scene({"versions": {**versions, "config_sha256": "source", "upstream_files_sha256": "source"},
                        "prefix": [], "parent_scene_id": "p", "split": "dev"})
    with pytest.raises(ValueError, match="version"):
        validate_scene(scene, versions, {}, diagnostic_code_drift=True)
    validate_scene(scene, versions, {}, diagnostic_implementation_drift=True)


def test_attempt_costs_can_be_summed_without_double_counting(local_dir):
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir/"dataset")
    config["logging"]["run_root"] = str(local_dir/"run")
    task = WrapperTask()
    wrapper = UniVTACRLWrapper(task, config, "cost", {"v": 1})
    costs = []
    for _ in range(2):
        wrapper.reset(0)
        wrapper.step(Action("recovery_delta", [0]*7))
        wrapper.stop()
        costs.append(json.loads(wrapper.logger.handle.attrs["outcome"])["episode_cost"])
    assert sum(c["initialization"] for c in costs) == task.ledger["initialization"] == 24
    assert sum(c["recovery"] for c in costs) == task.ledger["recovery"] == 12
    wrapper.close()


def test_parent_split_persists_across_wrappers_and_conditions(local_dir):
    from evotac.data.scene_registry import SceneRegistry
    first = SceneRegistry(local_dir, {"parent": "dev"})
    assert first.register("parent") == "dev"
    second = SceneRegistry(local_dir)
    with pytest.raises(ValueError, match="belongs"):
        second.register("parent", "test")
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir/"dataset")
    config["logging"]["run_root"] = str(local_dir/"run")
    wrapper = UniVTACRLWrapper(WrapperTask(), config, "conditions", {"v": 1})
    wrapper.reset(0)
    parent = wrapper.scene_metadata["parent_scene_id"]
    assert parent == "insert_hole:source_seed:0" and wrapper.scene_metadata["split"] == "dev"
    wrapper.reset(0, {"kind": "lateral_offset", "offset_world_m": [.002, 0]})
    assert wrapper.scene_metadata["parent_scene_id"] == parent
    with pytest.raises(ValueError, match="belongs"):
        wrapper.reset(0, split="test")
    wrapper.close()


def test_training_metadata_matches_post_action_success_check(local_dir):
    class SuccessMetadataTask(WrapperTask):
        def check_success(self):
            self.metadata["checked_physics"] = self._physics_step_count
            return True
    config = load_config()
    config["logging"]["dataset_root"] = str(local_dir/"dataset")
    config["logging"]["run_root"] = str(local_dir/"run")
    wrapper = UniVTACRLWrapper(SuccessMetadataTask(), config, "metadata", {"v": 1})
    wrapper.reset(0)
    obs, _, _, _, info = wrapper.step(Action("recovery_delta", [0]*7))
    assert info["training_info"]["task_metadata"]["checked_physics"] == obs["physics_step"]
    assert info["training_info"]["task_success"] is True
    assert "task_metadata" not in obs
    wrapper.close()


def test_flushed_prefix_reopens_after_forced_process_exit(local_dir):
    import subprocess
    import sys
    code = '''
import sys,time
from evotac.data.rollout_logger import RolloutLogger
logger=RolloutLogger(sys.argv[1], {"test_double": True}, flush_every=1)
logger.initial_observation({"value": 0}, {})
for i in range(3):
    logger.append({"physics_steps": 6}, {"value": i+1}, {"terminated": False}, {})
print("flushed", flush=True)
while True:
    time.sleep(1)
'''
    path = local_dir/"killed.h5"
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "flushed"
        child.kill()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    with h5py.File(path, "r") as handle:
        assert not handle.attrs["complete"]
        assert len(handle["observations"]) == 4
        assert len(handle["actions"]) == len(handle["transitions"]) == 3
        assert handle["observations/3/value"][()] == 3


def test_collection_interrupt_retains_attempt_denominator(local_dir, monkeypatch):
    from types import SimpleNamespace
    from evotac.scripts import collect_rollouts
    manifest = local_dir/"scenes.jsonl"
    manifest.write_text("\n".join(json.dumps({"seed": seed, "parent_scene_id": f"p{seed}",
                                            "split": "dev", "actions": []}) for seed in (0, 1)))
    args = SimpleNamespace(scene_manifest=str(manifest), run_id="interrupted", device="cpu")
    closed = []
    class InterruptedWrapper:
        def reset(self, *args, **kwargs):
            raise KeyboardInterrupt("collection interrupted during reset")
        def close(self):
            closed.append("wrapper")
    monkeypatch.setattr(collect_rollouts, "parser_for", lambda _: SimpleNamespace(
        add_argument=lambda *a, **kw: None, parse_args=lambda: args))
    monkeypatch.setattr(collect_rollouts, "launch", lambda _: (
        {"budgets": {"stop_on_exception": True}}, {"run_root": local_dir}, {},
        SimpleNamespace(app=SimpleNamespace(close=lambda: closed.append("app")))))
    monkeypatch.setattr(collect_rollouts, "build_wrapper", lambda *a: InterruptedWrapper())
    with pytest.raises(KeyboardInterrupt):
        collect_rollouts.main()
    checks = json.loads((local_dir/"checks.json").read_text())
    assert checks["attempted"] == 1 and checks["listed"] == 2
    assert "KeyboardInterrupt" in checks["outcomes"][0]["error"]
    assert closed == ["wrapper", "app"]
