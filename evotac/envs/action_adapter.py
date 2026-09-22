"""One-boundary actions: world-frame rotation vectors and local damped IK."""
from dataclasses import dataclass
import numpy as np

from evotac.data.schemas import Action, ActionKind, RobotState


class ActionRejected(ValueError):
    """A legal action could not be converted without violating execution limits."""


def quaternion_product(a, b):
    aw, av = a[0], np.asarray(a[1:])
    bw, bv = b[0], np.asarray(b[1:])
    return np.r_[aw * bw - av @ bv, aw * bv + bw * av + np.cross(av, bv)]


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ActionRejected("Invalid base quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def apply_world_rotvec(quaternion, vector):
    vector = np.asarray(vector, dtype=float)
    theta = np.linalg.norm(vector)
    delta = np.r_[np.cos(theta / 2), vector * (0.5 if theta < 1e-12 else np.sin(theta / 2) / theta)]
    result = quaternion_product(delta, quaternion)
    return result / np.linalg.norm(result)


@dataclass
class Command:
    joint_target: np.ndarray
    gripper_target: float
    details: dict

    def as_dict(self):
        return {"joint_target": self.joint_target.copy(), "gripper_target": self.gripper_target,
                "arm_velocity_target": np.zeros(7), "finger_velocity_target": np.zeros(2),
                "force": False, **self.details}


class ActionAdapter:
    def __init__(self, config):
        self.config = config

    def adapt(self, action: Action, state: RobotState):
        if not isinstance(action, Action):
            raise TypeError("Use an explicitly typed Action")
        c = self.config
        q = np.asarray(state.joint_position, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ActionRejected("Invalid measured arm state")
        values = np.array(action.values)
        details = {"kind": action.kind.value, "version": action.version}
        if action.kind is ActionKind.JOINT_TARGET:
            target, grip = values[:7], values[7]
        else:
            clipped = np.clip(values, -1, 1)
            world_delta = np.r_[clipped[:3] * c["translation_scale"], clipped[3:6] * c["rotation_scale"]]
            rotation = rotation_matrix(state.base_quaternion_world)
            frame = np.zeros((6, 6))
            frame[:3, :3] = frame[3:, 3:] = rotation.T
            jacobian = np.asarray(state.jacobian_world, dtype=float)
            if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
                raise ActionRejected("Missing/invalid panda_hand world Jacobian")
            jacobian = frame @ jacobian
            delta = frame @ world_delta
            singular = np.linalg.svd(jacobian, compute_uv=False)
            if singular[-1] < c["ik_min_singular_value"] and np.linalg.norm(delta) > 0:
                raise ActionRejected("IK singularity threshold exceeded")
            try:
                dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + c["ik_damping"]**2 * np.eye(6), delta)
            except np.linalg.LinAlgError as exc:
                raise ActionRejected("IK solve failed") from exc
            target = q + dq
            grip = float(np.mean(state.finger_position)) + clipped[6] * c["gripper_scale"]
            details.update(normalized_clipped=clipped, world_delta=world_delta,
                           base_delta=delta, ik_singular_values=singular,
                           desired_ee_position_world=state.ee_position_world + world_delta[:3],
                           desired_ee_quaternion_world=apply_world_rotvec(state.ee_quaternion_world, world_delta[3:]))
        lower = np.maximum(c["joint_lower"], q - c["max_joint_delta"])
        upper = np.minimum(c["joint_upper"], q + c["max_joint_delta"])
        if np.any(lower > upper):
            raise ActionRejected("Measured joints are too far outside legal range")
        limited = np.clip(target, lower, upper)
        limited_grip = float(np.clip(grip, *c["gripper_range"]))
        details.update(joint_unclipped=target.copy(), joint_clip_delta=limited - target,
                       gripper_unclipped=float(grip), gripper_clip_delta=limited_grip - grip)
        if action.kind is ActionKind.RECOVERY_DELTA:
            residual = jacobian @ (limited - q) - delta
            details["ik_residual_base"] = residual
            if np.linalg.norm(residual[:3]) > c["ik_max_translation_residual"] or np.linalg.norm(residual[3:]) > c["ik_max_rotation_residual"]:
                raise ActionRejected("IK residual after joint limiting exceeds threshold")
        if not np.isfinite(limited).all() or not np.isfinite(limited_grip):
            raise ActionRejected("Nonfinite command")
        return Command(limited, limited_grip, details)
