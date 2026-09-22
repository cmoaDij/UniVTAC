"""Isolated insert_hole extension.

pre_move stages follow envs/insert_hole.py at 85a047aadafc9e60dcd43705b79585068b30cf10.
Only the task-specific sequence is copied; BaseTask physics/render order is reused.
"""
import numpy as np
import torch

from envs.insert_hole import Task as OriginalTask, TaskCfg as OriginalTaskCfg
from envs._base_task import configclass, construct_grasp_pose
from evotac.data.schemas import RobotState, freeze
from evotac.envs.fixed_step_executor import targets_match_readback


@configclass
class TaskCfg(OriginalTaskCfg):
    pass


class EvoTacInsertHoleTask(OriginalTask):
    def __init__(self, cfg, evotac_config, **kwargs):
        self.evotac_config = evotac_config
        self.phase = "initialization"
        self.ledger = {name: 0 for name in ("initialization", "prefix", "stabilization", "recovery", "base")}
        self.prefix = []
        self.references = {}
        self.sample_steps = {}
        self.sample_frames = {}
        self.sensor_frames = {}
        self.replay_prefix = None
        self.record_prefix = False
        self.condition = {"kind": "nominal"}
        super().__init__(cfg, **kwargs)

    def read_robot_state(self):
        manager = self._robot_manager
        robot, data = manager.robot, manager.robot.data
        if not robot.is_fixed_base:
            raise RuntimeError("Phase 1 requires a fixed-base Franka")
        if manager._body_name != "panda_hand":
            raise RuntimeError("Jacobian control point mismatch")
        arms = manager._arm_ids
        jacobian = robot.root_physx_view.get_jacobians()[0, manager._body_idx-1, :, arms]
        return RobotState(
            freeze(data.joint_pos[0, arms]), freeze(data.joint_vel[0, arms]),
            freeze(data.joint_pos[0, manager._gripper_ids]),
            freeze(data.body_link_pos_w[0, manager._body_idx]),
            freeze(data.body_link_quat_w[0, manager._body_idx]),
            freeze(data.root_link_quat_w[0]), freeze(jacobian))

    def target_buffers(self):
        m = self._robot_manager
        return {"arm_position": freeze(m.robot.data.joint_pos_target[0, m._arm_ids]),
                "arm_velocity": freeze(m.robot.data.joint_vel_target[0, m._arm_ids]),
                "finger_position": freeze(m.robot.data.joint_pos_target[0, m._gripper_ids]),
                "finger_velocity": freeze(m.robot.data.joint_vel_target[0, m._gripper_ids]),
                "force": False}

    def set_targets(self, targets):
        m = self._robot_manager
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        m.set_arm(tensor(targets["arm_position"]), tensor(targets["arm_velocity"]), force=False)
        m.set_gripper(tensor(targets["finger_position"]), tensor(targets["finger_velocity"]), force=False)

    def apply_command(self, command):
        expected = {"arm_position": command.joint_target, "arm_velocity": np.zeros(7),
                    "finger_position": np.full(2, command.gripper_target), "finger_velocity": np.zeros(2)}
        self.set_targets(expected)
        accepted = self.target_buffers()
        confirmed = targets_match_readback(expected, accepted)
        return {"accepted": bool(confirmed), "acceptance_evidence": "articulation_target_buffer_readback", **accepted}

    def move(self, *args, **kwargs):
        kwargs["force"] = False
        kwargs["is_save"] = False
        return super().move(*args, **kwargs)

    def take_dense_action(self, control_seq, is_save=False, force=False):
        return super().take_dense_action(control_seq, is_save=False, force=False)

    def _step(self, is_save=False):
        start = self._physics_step_count
        event = None
        if self.record_prefix:
            event = {"kind": "hold", "targets": self.target_buffers(), "physics_steps": 0}
            self.prefix.append(event)
        try:
            return super()._step(is_save=False)
        finally:
            elapsed = self._physics_step_count - start
            self.ledger[self.phase] += elapsed
            if event is not None:
                event["physics_steps"] = elapsed

    def _update_render(self):
        super()._update_render()
        # Access data to complete lazy updates, then consult actual frame counters.
        # Re-rendering one physics state never advances the reported sample time.
        sensors = [("camera", n, s, ["rgb"]) for n, s in self._camera_manager.cameras.items()]
        sensors += [("tactile", n, t.sensor, ["rgb", "rgb_marker"]) for n, t in self._tactile_manager.tactiles.items()]
        for category, name, sensor, fields in sensors:
            _ = sensor.data
            frame = int(sensor.frame[0])
            key = f"{category}/{name}"
            if self.sensor_frames.get(key) != frame:
                for field in fields:
                    self.sample_steps[f"{key}/{field}"] = int(self._physics_step_count)
                    self.sample_frames[f"{key}/{field}"] = frame
                self.sensor_frames[key] = frame
        if self.record_prefix:
            self.prefix.append({"kind": "render", "physics_step": int(self._physics_step_count)})

    def sample_reference(self, kind, *, synchronize=True):
        if synchronize:
            self._update_render()
        images = freeze(self._tactile_manager.get_observations(["rgb", "rgb_marker"]))
        self.references[kind] = {"kind": kind, "images": images,
                                 "physics_step": int(self._physics_step_count),
                                 "valid": bool(self.plan_success)}
        if self.record_prefix:
            self.prefix.append({"kind": "reference", "reference_kind": kind})

    def reset(self, seed=-1, **kwargs):
        self.phase = "initialization"
        self.record_prefix = False
        self.prefix, self.references, self.sample_steps, self.sensor_frames = [], {}, {}, {}
        self.sample_frames = {}
        # The base reset zeros its counters. Ledger is cumulative and never reset.
        self._robot_manager.robot.set_joint_velocity_target(
            torch.zeros_like(self._robot_manager.robot.data.joint_vel))
        self.bootstrap_info = None
        if self.first_frame is None:
            # BaseTask's first reset saves a UIPC frame after five steps, while
            # subsequent resets restore it. Prime that standard frame without
            # grasping, then run the real attempt through the restore path too.
            # Every priming step is still charged to the initialization ledger.
            start_cost = sum(self.ledger.values())
            self.bootstrap_only = True
            try:
                super().reset(seed=seed, **kwargs)
            finally:
                self.bootstrap_only = False
            self.bootstrap_info = {"protocol": "standard_frame_before_prefix.v1",
                                   "physics_steps": sum(self.ledger.values())-start_cost}
            self.prefix, self.references, self.sample_steps, self.sensor_frames = [], {}, {}, {}
            self.sample_frames = {}
            self._robot_manager.robot.set_joint_velocity_target(
                torch.zeros_like(self._robot_manager.robot.data.joint_vel))
        result = super().reset(seed=seed, **kwargs)
        if not self.plan_success:
            raise RuntimeError("Physical initialization/prefix planning failed")
        self.phase = "stabilization"
        targets = self.target_buffers()
        targets["arm_velocity"] = np.zeros(7)
        targets["finger_velocity"] = np.zeros(2)
        self.set_targets(targets)
        for _ in range(self.evotac_config["budgets"]["stable_physics_steps"]):
            self._step(is_save=False)
        self.phase = "base"
        return result

    def _setup_insert_geometry(self):
        base_pose = self.slot.get_pose()
        base_pose[3:] = (1, 0, 0, 0)
        self.metadata["rotate"] = self.rotate
        self.hole_pose = base_pose.add_bias([0., 0., .1])
        self.target_pose = self.hole_pose.add_rotation([0, -np.pi/6 if self.rotate == 0 else np.pi/6, 0])

    def pre_move(self):
        if getattr(self, "bootstrap_only", False):
            return
        self.phase = "prefix"
        self.initial_prefix_state = self.read_robot_state().as_dict()
        self.initial_prefix_physics = int(self._physics_step_count)
        self._setup_insert_geometry()
        if self.replay_prefix is not None:
            for event in self.replay_prefix:
                if event["kind"] == "hold":
                    self.set_targets(event["targets"])
                    for _ in range(event["physics_steps"]):
                        self._step(is_save=False)
                elif event["kind"] == "render":
                    self._update_render()
                elif event["kind"] == "reference":
                    self.sample_reference(event["reference_kind"], synchronize=False)
                else:
                    raise ValueError(f"Unknown prefix event: {event['kind']}")
            self.prefix = freeze(self.replay_prefix)
        else:
            self.record_prefix = True
            try:
                self.delay(10)
                self.sample_reference("empty")
                bias = self.rng.uniform(.095, .10)
                target = self.prism.get_pose().add_bias([0, 0, bias])
                cpose = construct_grasp_pose(target.p, [0, 0, 1], [1, 0, 0])
                self.cid = self.prism.register_point(cpose, type="contact")
                self.move(self.atom.grasp_actor(self.prism, contact_point_id=self.cid, pre_dis=0., dis=0.))
                self.origin_inhand_pose = self.prism.get_pose().rebase(self._robot_manager.get_gripper_center_pose())
                self.move(self.atom.move_by_displacement(z=.15), constraint_pose=[1, 1, 1, 1, 1, 0])
                stable_targets = self.target_buffers()
                stable_targets["arm_velocity"] = np.zeros(7)
                stable_targets["finger_velocity"] = np.zeros(2)
                self.set_targets(stable_targets)
                self.delay(self.evotac_config["budgets"]["stable_physics_steps"])
                if self.check_early_stop():
                    raise RuntimeError("Physical grasp was lost before the grasp reference")
                self.sample_reference("grasp")  # lifted, before approaching external hole geometry
                self.move(self.atom.place_actor(self.prism, target_pose=self.hole_pose, pre_dis=.05, dis=.01, is_open=False))
                self.move(self.atom.place_actor(self.prism, target_pose=self.hole_pose, pre_dis=.01, dis=.002, is_open=False), constraint_pose=[1, 1, 1, 1, 1, 0])
                kind = self.condition.get("kind", "nominal")
                if kind == "lateral_offset":
                    offset = np.asarray(self.condition["offset_world_m"], dtype=float)
                    if offset.shape != (2,) or not np.isfinite(offset).all() or np.max(np.abs(offset)) > .02:
                        raise ValueError("Lateral offset must have two finite values within 0.02 m")
                    self.move(self.atom.move_by_displacement(x=float(offset[0]), y=float(offset[1])))
                elif kind != "nominal":
                    raise ValueError(f"Unsupported condition {kind}")
            finally:
                self.record_prefix = False
        # Verify source grasp quality before replacing the in-hand baseline.
        # Replayed references and object state are checked by reconstruction_report.
        if self.replay_prefix is None and self.check_early_stop():
            raise RuntimeError("Physical grasp was lost during approach")
        self.origin_inhand_pose = self.prism.get_pose().rebase(self._robot_manager.get_gripper_center_pose())
        self.phase = "stabilization"
