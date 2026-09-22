"""Pure configuration loading; all project outputs are relative to evotac/."""
from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def output_path(path):
    path = Path(path)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError(f"Output must be inside {ROOT}: {path}")
    if resolved.parts[len(ROOT.parts)] not in {"runs", "datasets", ".cache", "checkpoints"}:
        raise ValueError(f"Output conflicts with project source: {resolved}")
    return resolved


def load_config(path=None):
    path = ROOT / "configs/phase1_insert_hole.yaml" if path is None else Path(path)
    config = yaml.safe_load(path.read_text())
    return validate_config(config)


def validate_config(config):
    config = deepcopy(config)
    sim = config["simulation"]
    physical, control = sim["physical_hz"], sim["control_hz"]
    if (isinstance(physical, bool) or isinstance(control, bool)
            or not isinstance(physical, int) or not isinstance(control, int)
            or physical <= 0 or control <= 0 or physical % control):
        raise ValueError("Frequencies must be positive integers with an integral ratio")
    if (physical, control, sim["num_envs"], sim["backend"], sim["sensor"], sim["enable_cameras"]) != (120, 20, 1, "taxim", "gsmini", True):
        raise ValueError("Phase 1 requires 120/20 Hz, one gsmini/taxim environment and cameras")
    c = config["controller"]
    if c["force"] is not False or c["control_point"] != "panda_hand":
        raise ValueError("Phase 1 requires force=False and panda_hand control point")
    from evotac.data.schemas import ACTION_VERSION, PREFIX_VERSION
    if config["replay"]["protocol_version"] != PREFIX_VERSION:
        raise ValueError("Unsupported prefix/initialization protocol")
    if c["action_version"] != ACTION_VERSION:
        raise ValueError("Unsupported action contract")
    for key, size in (("joint_lower", 7), ("joint_upper", 7), ("translation_scale", 3), ("rotation_scale", 3), ("gripper_range", 2)):
        values = np.asarray(c[key], dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid controller.{key}")
    if np.any(np.asarray(c["joint_lower"]) >= c["joint_upper"]) or c["gripper_range"][0] >= c["gripper_range"][1]:
        raise ValueError("Invalid joint limits")
    for key in ("arm_stiffness", "max_joint_delta", "gripper_scale", "ik_damping", "ik_min_singular_value", "ik_max_translation_residual", "ik_max_rotation_residual"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"controller.{key} must be positive and finite")
    if not np.isfinite(c["arm_damping"]) or c["arm_damping"] < 0:
        raise ValueError("arm_damping must be nonnegative and finite")
    if min(*c["translation_scale"], *c["rotation_scale"]) <= 0:
        raise ValueError("Action scales must be positive")
    reset_wall = config["budgets"]["reset_wall_seconds"]
    if not np.isfinite(reset_wall) or reset_wall <= 0:
        raise ValueError("Reset wall-clock guard must be positive and finite")
    for key in ("task_physics_steps", "recovery_physics_steps", "stable_physics_steps"):
        value = config["budgets"][key]
        if type(value) is not int or value <= 0 or value % (physical // control):
            raise ValueError(f"budgets.{key} must be a positive multiple of decimation")
    if config["budgets"]["recovery_physics_steps"] > config["budgets"]["task_physics_steps"]:
        raise ValueError("Recovery budget is a subset of task budget")
    if config["observation"]["history_length"] < 8 or config["observation"]["max_frame_age_steps"] < 0:
        raise ValueError("Invalid observation history or frame age")
    if config["logging"]["flush_every"] < 1 or config["logging"]["image_storage"] != "lossless_gzip":
        raise ValueError("Invalid logging configuration")
    paths = [output_path(config["logging"][key]) for key in ("dataset_root", "run_root", "cache_root")]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Output roots must not overlap")
    return config


def run_paths(config, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Invalid run_id")
    return {key: output_path(config["logging"][key]) / run_id for key in ("dataset_root", "run_root")}
