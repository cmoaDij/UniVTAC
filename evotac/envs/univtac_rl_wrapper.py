"""Single-environment API. Records every attempt before calling task.reset."""
import json
import uuid
from copy import deepcopy
import numpy as np
import h5py
from pathlib import Path

from evotac.config import ROOT, run_paths
from evotac.data.schemas import Action, ActionKind, freeze, json_value
from evotac.data.scene_registry import SceneRegistry
from evotac.data.legacy_demonstrations import LegacyDemonstrations, load_legacy_config
from evotac.data.rollout_logger import RolloutLogger, write_tree, read_tree
from evotac.envs.action_adapter import ActionAdapter, Command
from evotac.envs.fixed_step_executor import FixedStepExecutor
from evotac.envs.observation_contract import ObservationContract
from evotac.envs.state_replay import PolicyState, seal_scene, validate_scene, reconstruction_report, digest, execution_fingerprint


class UniVTACRLWrapper:
    def __init__(self, task, config, run_id, versions):
        self.task, self.config, self.versions = task, config, versions
        self.paths = run_paths(config, run_id)
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        simulation = config["simulation"]
        physical_hz, control_hz = int(simulation["physical_hz"]), int(simulation["control_hz"])
        if physical_hz % control_hz:
            raise ValueError("simulation physical_hz must be divisible by control_hz")
        self.executor = FixedStepExecutor(task, ActionAdapter(config["controller"]),
                                          physical_hz=physical_hz,
                                          decimation=physical_hz // control_hz)
        self.contract = ObservationContract(config["observation"], physical_hz=int(simulation["physical_hz"]))
        self.policy_state = PolicyState(config["observation"]["history_length"])
        self.logger = None
        self.done = True
        legacy = LegacyDemonstrations(load_legacy_config(ROOT/config["legacy_data"]["config"]))
        source_splits = {row["parent_scene_id"]: row["split"] for row in legacy.manifest()}
        self.registry = SceneRegistry(config["logging"]["dataset_root"], source_splits)
        self.split_by_parent = dict(source_splits)
        self.last_observation = None
        self.scene_metadata = {}
        self.task_elapsed = self.recovery_elapsed = 0
        self.control_prefix = []

    def _write_jsonl(self, path, value):
        with path.open("a") as handle:
            handle.write(json.dumps(json_value(value), allow_nan=False)+"\n")

    def _observe(self):
        if self.task._last_render_physics_step != self.task._physics_step_count:
            self.task._update_render()
        raw = self.task._get_observations()
        remaining = self.config["budgets"]["task_physics_steps"]-self.task_elapsed
        obs = self.contract.build(raw, self.task.read_robot_state(), self.task._physics_step_count,
                                  self.task.sample_steps, self.task.references, remaining, getattr(self.task, "sample_frames", {}))
        obs["remaining_recovery_steps"] = max(0, self.config["budgets"]["recovery_physics_steps"] - self.recovery_elapsed)
        return obs, {"actor": freeze(raw.get("actor", {})), "task_metadata": freeze(self.task.metadata)}

    def reset(self, seed, condition=None, *, parent_scene_id=None, split=None, branch_id="base"):
        if self.logger is not None:
            if not self.done:
                self.stop("reset_interrupted")
            self.logger.close()
        condition = condition or {"kind": "nominal"}
        if type(seed) is not int or seed < 0:
            raise ValueError("Explicit nonnegative seed required")
        parent = parent_scene_id or f"insert_hole:source_seed:{seed}"
        split = self.registry.register(parent, split)
        if parent in self.split_by_parent and self.split_by_parent[parent] != split:
            raise ValueError("Branches of one parent must share a split")
        self.split_by_parent[parent] = split
        self.scene_metadata = {"parent_scene_id": parent, "episode_id": uuid.uuid4().hex,
                               "branch_id": branch_id, "seed": seed, "split": split,
                               "condition": freeze(condition), "versions": self.versions,
                               "ledger_at_attempt_start": freeze(self.task.ledger)}
        self.logger = RolloutLogger(self.paths["dataset_root"] / "episodes" / f'{self.scene_metadata["episode_id"]}.h5', self.scene_metadata, self.config["logging"]["flush_every"])
        self._write_jsonl(self.paths["dataset_root"] / "scenes.jsonl", {**self.scene_metadata, "event": "attempt_started"})
        self.done = False
        self.contract.reset()
        self.policy_state.reset()
        self.control_prefix = []
        self.task_elapsed = self.recovery_elapsed = 0
        self.task.condition = freeze(condition)
        try:
            self.task.reset(seed=seed)
            self.last_observation, training = self._observe()
            if any(not modality["valid"] for group in self.last_observation["images"].values()
                   for sensor in group.values() for modality in sensor.values()):
                raise RuntimeError("Reset returned missing, stale or invalid sensor frames")
            self.task_start_physics = int(self.task._physics_step_count)
            self.logger.initial_observation(self.last_observation, self.task.references)
            self.logger.event("reset_complete", {"prefix": self.task.prefix, "initial_prefix_state": self.task.initial_prefix_state,
                                                  "ledger": self.task.ledger, "physics_steps": sum(self._episode_cost().values()),
                                                  "bootstrap": getattr(self.task, "bootstrap_info", None),
                                                  "decision_start_physics": self.task_start_physics})
            return self.last_observation, {"execution_info": {"reset_physics_steps": sum(self._episode_cost().values()),
                                                               "decision_start_physics": self.task_start_physics,
                                                               "ledger": freeze(self.task.ledger)},
                                           "training_info": training, "scene_metadata": self.scene_metadata}
        except BaseException as exc:
            outcome = {"reason": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "reset_exception",
                       "valid_trial": False, "terminated": False, "truncated": False,
                       "bootstrap_allowed": False, "incomplete": True,
                       "exception": f"{type(exc).__name__}: {exc}", "ledger": freeze(self.task.ledger),
                       "episode_cost": self._episode_cost()}
            self.logger.event("reset_exception", {**outcome, "partial_prefix": self.task.prefix})
            self.logger.finish(outcome)
            self.done = True
            raise

    def step(self, action):
        return self._step(action)

    def _step(self, action, *, recorded_command=None):
        if self.done:
            raise RuntimeError("Episode ended; reset before step")
        if not isinstance(action, Action):
            raise TypeError("step requires an explicitly typed Action")
        recovery = action.kind is ActionKind.RECOVERY_DELTA
        budgets = self.config["budgets"]
        if budgets["task_physics_steps"]-self.task_elapsed < 6 or (recovery and budgets["recovery_physics_steps"]-self.recovery_elapsed < 6):
            outcome = self._finish("recovery_budget" if recovery and budgets["recovery_physics_steps"]-self.recovery_elapsed < 6 else "task_budget", False, True, True)
            return self.last_observation, 0., False, True, {"execution_info": {"physics_steps": 0, **outcome}, "training_info": {}, "scene_metadata": self.scene_metadata}
        self.task.phase = "recovery" if recovery else "base"
        try:
            execution = self.executor.execute(action, recorded_command=recorded_command)
        except BaseException as exc:
            self.stop("interrupted")
            raise
        self.task_elapsed += execution["physics_steps"]
        if recovery:
            self.recovery_elapsed += execution["physics_steps"]
        training, next_obs = {}, None
        try:
            if execution["valid_trial"]:
                self.contract.previous_action = freeze(execution["applied_action"])
                next_obs, training = self._observe()
                # The returned next observation includes the just-completed
                # transition in its history. accept() below commits that same
                # entry after the transition has been assembled.
                history_count = min(self.contract.history.maxlen, len(self.contract.history)+1)
                next_obs["history_mask"] = ([False]*(self.contract.history.maxlen-history_count)
                                            + [True]*history_count)
                if any(not m["valid"] for group in next_obs["images"].values() for sensor in group.values() for m in sensor.values()):
                    execution.update(valid_trial=False, status="infrastructure_error", reason="invalid_sensor_frame")
            success = bool(self.task.check_success()) if execution["valid_trial"] else False
            failure = bool(self.task.check_early_stop()) if execution["valid_trial"] and not success else False
        except Exception as exc:
            execution.update(valid_trial=False, status="infrastructure_error", reason=f"observation:{type(exc).__name__}: {exc}")
            success = failure = False
        if execution["valid_trial"]:
            training.update(task_metadata=freeze(self.task.metadata), task_success=success,
                            object_lost=failure, condition=freeze(self.task.condition))
        execution["execution_reason"] = execution.get("reason")
        execution["last_valid_physics_step"] = self.last_observation["physics_step"]
        terminated, truncated = False, False
        reason = "running"
        if not execution["valid_trial"]:
            reason, truncated = "infrastructure_exception", True
        elif execution["status"] != "executed":
            reason, terminated = "execution_rejected", True
        elif success or failure:
            reason, terminated = ("success" if success else "object_lost"), True
        elif self.task_elapsed >= budgets["task_physics_steps"]:
            reason, truncated = "task_budget", True
        elif recovery and self.recovery_elapsed >= budgets["recovery_physics_steps"]:
            reason, truncated = "recovery_budget", True
        outcome = {"reason": reason, "terminated": terminated, "truncated": truncated,
                   "valid_trial": execution["valid_trial"], "incomplete": not execution["valid_trial"],
                   "bootstrap_allowed": reason == "running",
                   "reward": float(success), "reward_components": {"task_success": float(success)}}
        self.logger.append(execution, next_obs, outcome, training)
        self.contract.accept(self.last_observation, action.as_dict())
        self.policy_state.rebuild(list(self.contract.history))
        self.control_prefix.append({"action": action.as_dict(), "execution": freeze(execution)})
        if next_obs is not None:
            self.last_observation = next_obs
        if reason != "running":
            self.logger.finish({**outcome, "ledger": self.task.ledger, "episode_cost": self._episode_cost()})
            self.done = True
        return next_obs, float(success), terminated, truncated, {"execution_info": {**execution, **outcome, "ledger": freeze(self.task.ledger)}, "training_info": training, "scene_metadata": freeze(self.scene_metadata)}

    def _episode_cost(self):
        start = self.scene_metadata["ledger_at_attempt_start"]
        return {phase: count-start.get(phase, 0) for phase, count in self.task.ledger.items()}

    def snapshot_shared_state(self):
        """Freeze the live simulator state for strict paired branches.

        Prefix replay reconstructs the visible prefix but can miss hidden UIPC
        FEM/contact state.  This snapshot asks UIPC to dump its current frame,
        records the PhysX robot state and all experiment counters, then returns
        a digestable in-memory handle.  It is intentionally process-local:
        restoring it in a fresh Isaac process is not certified.
        """
        if self.logger is None or self.done:
            raise RuntimeError("a live episode is required for a shared-state snapshot")
        task = self.task
        # Sensor rendering consumes the TacEx marker RNG. Save its state
        # before the synchronized source render so a restored branch produces
        # the same first marker sample as the source observation.
        marker_rng_states = {}
        for name, tactile in task._tactile_manager.tactiles.items():
            simulator = getattr(tactile.sensor, "marker_motion_simulator", None)
            marker_sim = getattr(simulator, "marker_motion_sim", None)
            marker_rng = getattr(marker_sim, "_marker_rng", None)
            if marker_rng is not None:
                marker_rng_states[name] = deepcopy(marker_rng.bit_generator.state)
        # The source branch must use the same render synchronization protocol
        # as a restored branch.  Without this explicit pass, export_scene can
        # retain the previous camera buffer while restore_shared_state reads a
        # freshly submitted RTX frame, making renderer latency look like state
        # divergence.
        task._update_render()
        self.last_observation, _ = self._observe()
        robot = task._robot_manager.robot
        task.uipc_sim.save_frame()
        joint_pos = robot.data.joint_pos.detach().cpu().numpy().copy()
        joint_vel = robot.data.joint_vel.detach().cpu().numpy().copy()
        joint_target = robot.data.joint_pos_target.detach().cpu().numpy().copy()
        velocity_target = robot.data.joint_vel_target.detach().cpu().numpy().copy()
        actor_poses = {name: actor.get_pose("matrix").copy()
                       for name, actor in task._actor_manager.actors.items()}
        payload = {
            "uipc_frame": int(task.uipc_sim.world.frame()),
            "joint_pos": joint_pos, "joint_vel": joint_vel,
            "joint_target": joint_target, "velocity_target": velocity_target,
            "actor_poses": actor_poses,
            "marker_rng_states": marker_rng_states,
            "task_elapsed": int(self.task_elapsed), "recovery_elapsed": int(self.recovery_elapsed),
            "physics_step": int(task._physics_step_count),
            "phase": str(task.phase), "task_rng": deepcopy(task.rng.bit_generator.state),
            "contract_history": freeze(list(self.contract.history)),
            "contract_previous_action": freeze(self.contract.previous_action),
            "observation": freeze(self.last_observation),
            "references": freeze(task.references),
            "scene_metadata": freeze(self.scene_metadata),
        }
        payload["snapshot_digest"] = digest({
            "uipc_frame": payload["uipc_frame"], "joint_pos": joint_pos,
            "joint_vel": joint_vel, "actor_poses": actor_poses,
            "marker_rng_states": marker_rng_states,
            "task_elapsed": payload["task_elapsed"], "physics_step": payload["physics_step"]})
        return payload

    def restore_shared_state(self, snapshot, *, branch_id):
        """Restore a process-local snapshot and open a fresh branch logger."""
        if not isinstance(snapshot, dict) or "snapshot_digest" not in snapshot:
            raise ValueError("invalid shared-state snapshot")
        task = self.task
        if int(task.uipc_sim.world.frame()) < int(snapshot["uipc_frame"]):
            raise ValueError("snapshot frame is not available in the current UIPC workspace")
        if self.logger is not None:
            if not self.done:
                self.logger.finish({"reason": "shared_state_branch", "valid_trial": True,
                                    "terminated": False, "truncated": False, "incomplete": False,
                                    "bootstrap_allowed": False, "ledger": self.task.ledger,
                                    "episode_cost": self._episode_cost()})
            self.logger.close()
        if not task.uipc_sim.world.recover(int(snapshot["uipc_frame"])):
            raise RuntimeError("UIPC could not recover the shared snapshot frame")
        task.uipc_sim.world.retrieve()
        robot = task._robot_manager.robot
        joint_pos = np.asarray(snapshot["joint_pos"])
        joint_vel = np.asarray(snapshot["joint_vel"])
        import torch
        robot.write_joint_state_to_sim(torch.as_tensor(joint_pos, device=task.device),
                                       torch.as_tensor(joint_vel, device=task.device))
        robot.set_joint_position_target(torch.as_tensor(snapshot["joint_target"], device=task.device))
        robot.set_joint_velocity_target(torch.as_tensor(snapshot["velocity_target"], device=task.device))
        task.rng.bit_generator.state = deepcopy(snapshot["task_rng"])
        for name, state in snapshot.get("marker_rng_states", {}).items():
            tactile = task._tactile_manager.tactiles.get(name)
            simulator = getattr(tactile.sensor, "marker_motion_simulator", None) if tactile else None
            marker_sim = getattr(simulator, "marker_motion_sim", None)
            marker_rng = getattr(marker_sim, "_marker_rng", None)
            if marker_rng is not None:
                marker_rng.bit_generator.state = deepcopy(state)
        task.phase = snapshot["phase"]
        task._physics_step_count = int(snapshot["physics_step"])
        task._last_render_physics_step = -1
        self.task_elapsed = int(snapshot["task_elapsed"])
        self.recovery_elapsed = int(snapshot["recovery_elapsed"])
        self.contract.reset()
        self.contract.history.extend(deepcopy(snapshot["contract_history"])[-self.contract.history.maxlen:])
        self.contract.previous_action = deepcopy(snapshot.get("contract_previous_action"))
        self.policy_state.rebuild(list(self.contract.history))
        self.scene_metadata = {**deepcopy(snapshot["scene_metadata"]), "branch_id": branch_id,
                               "episode_id": uuid.uuid4().hex}
        self.logger = RolloutLogger(self.paths["dataset_root"] / "episodes" /
                                    f'{self.scene_metadata["episode_id"]}.h5',
                                    self.scene_metadata, self.config["logging"]["flush_every"])
        self.done = False
        self.control_prefix = []
        # Measure the restored simulator. Reusing source pixels would make
        # the observation audit tautological and hide rendering/state drift.
        self._update_observation_after_restore()
        self.logger.initial_observation(self.last_observation, task.references)
        self.logger.event("shared_state_restore", {"branch_id": branch_id,
                                                     "snapshot_digest": snapshot["snapshot_digest"],
                                                     "uipc_frame": snapshot["uipc_frame"]})
        return self.last_observation

    def _update_observation_after_restore(self):
        self.last_observation, _ = self._observe()
        return self.last_observation

    def _finish(self, reason, terminated=False, truncated=False, valid_trial=False):
        outcome = {"reason": reason, "terminated": terminated, "truncated": truncated,
                   "valid_trial": valid_trial, "incomplete": not (terminated or truncated),
                   "bootstrap_allowed": False, "ledger": freeze(self.task.ledger),
                   "episode_cost": self._episode_cost()}
        self.logger.finish(outcome)
        self.done = True
        return outcome

    def stop(self, reason="user_stop"):
        self.policy_state.switch_control()
        if self.logger is not None and not self.done:
            self.logger.event("stop", {"reason": reason, "incomplete": True})
            self._finish(reason)

    def export_scene(self):
        if self.done:
            raise RuntimeError("Only a live recovery start can be exported")
        obs, training = self._observe()
        scene = seal_scene({**self.scene_metadata, "prefix": self.task.prefix,
                            "initial_prefix_state": self.task.initial_prefix_state,
                            "initial_prefix_physics": self.task.initial_prefix_physics,
                            "task_start_physics": self.task_start_physics,
                            "observation": obs, "references": self.task.references, "training_info": training,
                            "history": list(self.contract.history), "policy_state": self.policy_state.export(),
                            "control_prefix": self.control_prefix, "task_elapsed_steps": self.task_elapsed,
                            "recovery_elapsed_steps": self.recovery_elapsed,
                            "remaining_task_steps": self.config["budgets"]["task_physics_steps"]-self.task_elapsed,
                            "rng_protocol": "numpy_default_rng_seed_before_reset.v1",
                            "stable_physics_steps": self.config["budgets"]["stable_physics_steps"]})
        scene_path = self.paths["dataset_root"] / "scene_refs" / (uuid.uuid4().hex+".h5")
        scene_path.parent.mkdir(exist_ok=True)
        with h5py.File(scene_path, "x") as handle:
            write_tree(handle, scene)
        self._write_jsonl(self.paths["dataset_root"] / "scenes.jsonl", {
            "event": "scene_exported", "scene_ref_path": str(scene_path),
            "parent_scene_id": scene["parent_scene_id"], "split": scene["split"],
            "integrity_sha256": scene["integrity_sha256"]})
        return scene

    def replay_start(self, scene_ref, *, diagnostic_code_drift=False,
                     diagnostic_implementation_drift=False):
        if isinstance(scene_ref, (str, Path)):
            with h5py.File(scene_ref, "r") as handle:
                scene_ref = read_tree(handle)
        try:
            validate_scene(scene_ref, self.versions, self.split_by_parent,
                           diagnostic_code_drift=diagnostic_code_drift,
                           diagnostic_implementation_drift=diagnostic_implementation_drift)
        except (ValueError, KeyError, TypeError) as exc:
            rejection = {"status": "incompatible", "valid_match": False, "physics_steps": 0,
                         "parent_scene_id": scene_ref.get("parent_scene_id"),
                         "reason": f"{type(exc).__name__}: {exc}"}
            self._write_jsonl(self.paths["run_root"] / "reconstruction.jsonl", rejection)
            if self.logger is not None and not self.done:
                self.logger.event("replay_rejected", rejection)
            raise
        tolerances = None
        tolerance_path = self.config["replay"]["tolerance_file"]
        if tolerance_path:
            tolerances = json.loads((ROOT / tolerance_path).read_text())
            if tolerances["version"] != self.config["replay"]["tolerance_version"]:
                raise ValueError("Tolerance version mismatch")
            if tolerances.get("execution_fingerprint") != execution_fingerprint(self.config):
                raise ValueError("Tolerance calibration does not match the execution configuration")
        self.task.replay_prefix = scene_ref["prefix"]
        try:
            self.reset(scene_ref["seed"], scene_ref["condition"], parent_scene_id=scene_ref["parent_scene_id"], split=scene_ref["split"], branch_id="replay_"+uuid.uuid4().hex[:8])
        finally:
            self.task.replay_prefix = None
        # The recorded joint commands, not a new IK solution, own the replay.
        # Replay execution is implemented by the same fixed-step entry point.
        for record in scene_ref["control_prefix"]:
            command = record["execution"]["commanded_action"]
            if record["execution"]["status"] != "executed":
                raise ValueError("Recovery starts cannot contain rejected prefix controls")
            action = Action(**record["action"])
            target = np.asarray(command["joint_target"], dtype=float)
            limits = self.config["controller"]
            grip = command["gripper_target"]
            if (target.shape != (7,) or not np.isfinite(target).all()
                    or np.any(target < limits["joint_lower"]) or np.any(target > limits["joint_upper"])
                    or not limits["gripper_range"][0] <= grip <= limits["gripper_range"][1]):
                raise ValueError("Recorded control target violates limits")
            restored = Command(target, grip, {"replayed_accepted_target": True,
                                               "original_command": freeze(command)})
            _, _, _, _, info = self._step(action, recorded_command=restored)
            if self.done:
                raise RuntimeError(f"Replay ended before start: {info['execution_info']['reason']}")
        self.recovery_elapsed = scene_ref["recovery_elapsed_steps"]
        self.policy_state.rebuild(list(self.contract.history))
        self.policy_state.switch_control()
        obs, training = self._observe()
        self.last_observation = obs
        report = reconstruction_report(scene_ref, obs, self.task.references, training,
                                       self.task.initial_prefix_state, list(self.contract.history), self.task_elapsed, tolerances,
                                       task_start_physics=self.task_start_physics, initial_prefix_physics=self.task.initial_prefix_physics)
        report["observable_match"] = report["valid_match"]
        report["versions_match"] = scene_ref["versions"] == self.versions
        report["source_versions"] = freeze(scene_ref["versions"])
        report["execution_versions"] = freeze(self.versions)
        if not report["versions_match"]:
            report["valid_match"] = False
            report["status"] = ("diagnostic_implementation_drift"
                                 if diagnostic_implementation_drift else "diagnostic_code_drift")
        report["ledger"] = freeze(self.task.ledger)
        report["initial_prefix_physics"] = self.task.initial_prefix_physics
        report["original_initial_prefix_physics"] = scene_ref["initial_prefix_physics"]
        self._write_jsonl(self.paths["run_root"] / "reconstruction.jsonl", report)
        self.logger.event("reconstruction", report)
        return obs, report

    def close(self):
        try:
            self.stop("closed")
            if self.logger is not None:
                self.logger.close()
        finally:
            self.task.close()
