"""Prefix reconstruction contracts, policy cache handoff and measured tolerances."""
import hashlib
import json
from collections import deque

import numpy as np
from evotac.data.schemas import freeze, json_value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def scene_digest(value):
    """Hash typed arrays directly, preserving the HDF5 tree representation.

    Numeric sequences are written as arrays by write_tree. Normalize them here
    too so an HDF5 round trip preserves the digest without expanding images
    into millions of Python objects. Length prefixes make the encoding
    unambiguous; byte order and memory layout are canonicalized.
    """
    hasher = hashlib.sha256(b"evotac.scene_binary.v1\0")

    def field(tag, data):
        hasher.update(tag)
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)

    def visit(item):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, np.ndarray) and item.ndim == 0:
            item = item.item()
        if isinstance(item, (list, tuple)) and item:
            try:
                array = np.asarray(item)
            except (ValueError, TypeError):
                array = None
            if array is not None and array.dtype.kind in "biuf":
                item = array
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in "biuf" or (item.dtype.kind == "f" and not np.isfinite(item).all()):
                raise ValueError("Scene arrays must contain finite numeric values")
            array = np.ascontiguousarray(item, dtype=item.dtype.newbyteorder("<"))
            field(b"A", json.dumps([array.dtype.str, array.shape]).encode())
            field(b"B", memoryview(array.reshape(-1)).cast("B"))
        elif isinstance(item, dict):
            field(b"D", str(len(item)).encode())
            for key in sorted(item):
                if not isinstance(key, str):
                    raise TypeError("Scene keys must be strings")
                field(b"K", key.encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            field(b"L", str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            field(b"S", json.dumps(item, sort_keys=True, allow_nan=False).encode())

    visit(value)
    return hasher.hexdigest()


def execution_fingerprint(config):
    return digest({**{k: config[k] for k in ("simulation", "controller", "observation", "budgets")},
                   "prefix_protocol": config["replay"]["protocol_version"]})


class PolicyState:
    """Phase-2 hook. Never re-import queued base actions at a handoff."""
    def __init__(self, history_length=8):
        self.history_length = history_length
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.history_length)
        self.action_queue = deque()

    def export(self):
        return {"history": freeze(list(self.history)), "history_length": self.history_length}

    def import_state(self, state):
        self.reset()
        self.history.extend(freeze(state["history"][-self.history_length:]))

    def rebuild(self, history):
        self.import_state({"history": history})

    def switch_control(self):
        self.action_queue.clear()


def seal_scene(scene):
    scene = freeze(scene)
    scene.pop("integrity_sha256", None)
    scene["integrity_algorithm"] = "evotac.scene_binary.v1"
    scene["integrity_sha256"] = scene_digest(scene)
    return scene


def validate_scene(scene, versions, split_by_parent, *, diagnostic_code_drift=False,
                   diagnostic_implementation_drift=False):
    payload = {k: v for k, v in scene.items() if k != "integrity_sha256"}
    algorithm = payload.get("integrity_algorithm")
    if algorithm == "evotac.scene_binary.v1":
        expected = scene_digest(payload)
    elif algorithm is None:
        expected = digest(payload)  # historical JSON seals remain verifiable
    else:
        raise ValueError(f"Unsupported scene integrity algorithm: {algorithm}")
    if scene.get("integrity_sha256") != expected:
        raise ValueError("Scene prefix or metadata integrity mismatch")
    source_versions = scene["versions"]
    version_diff = source_versions.keys() ^ versions.keys()
    version_diff |= {key for key in source_versions.keys() & versions.keys()
                     if source_versions[key] != versions[key]}
    allowed_drift = set()
    if diagnostic_code_drift:
        allowed_drift.add("evotac_code_sha256")
    if diagnostic_implementation_drift:
        # Implementation diagnostics may intentionally use a validation config
        # whose replay tolerance file/version differs from the source rollout.
        # The execution fingerprint is checked separately when the tolerance
        # file is loaded, so permitting this metadata hash here does not relax
        # the physical comparison itself.
        allowed_drift.update(("config_sha256", "evotac_code_sha256", "upstream_files_sha256"))
    if version_diff and not version_diff.issubset(allowed_drift):
        raise ValueError("Incompatible environment/controller/adapter version")
    parent, split = scene["parent_scene_id"], scene["split"]
    if parent in split_by_parent and split_by_parent[parent] != split:
        raise ValueError("Parent scene split mismatch")
    for event in scene["prefix"]:
        if event["kind"] not in {"hold", "render", "reference"}:
            raise ValueError("Unknown prefix event")
        if event["kind"] == "reference" and event["reference_kind"] not in {"empty", "grasp"}:
            raise ValueError("Unknown prefix reference")
        if event["kind"] == "hold":
            steps = event["physics_steps"]
            if (not isinstance(steps, (int, np.integer)) or isinstance(steps, (bool, np.bool_))
                    or steps < 0 or event["targets"].get("force") is not False):
                raise ValueError("Invalid prefix hold")
            for name, size in (("arm_position", 7), ("arm_velocity", 7), ("finger_position", 2), ("finger_velocity", 2)):
                values = np.asarray(event["targets"].get(name), dtype=float)
                if values.shape != (size,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid recorded target: {name}")


def array_error(left, right):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"max_abs": None, "rmse": None, "compatible": False}
    delta = a - b
    return {"max_abs": float(np.max(np.abs(delta))) if delta.size else 0.,
            "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0., "compatible": True}


def reconstruction_report(scene, observation, references, training_info, initial_state,
                          history, task_elapsed, tolerances=None, *, task_start_physics=None, initial_prefix_physics=None):
    original = scene["observation"]
    errors = {}
    for key in ("joint_position", "finger_position", "ee_position_world"):
        errors[key] = array_error(original["proprioception"][key], observation["proprioception"][key])
    q1 = np.asarray(original["proprioception"]["ee_quaternion_world"])
    q2 = np.asarray(observation["proprioception"]["ee_quaternion_world"])
    errors["ee_quaternion_world"] = array_error(q1, q2 if q1 @ q2 >= 0 else -q2)
    for name in ("left_tactile", "right_tactile"):
        for field in ("rgb", "rgb_marker"):
            errors[f"tactile.{name}.{field}"] = array_error(original["images"]["tactile"][name][field]["data"], observation["images"]["tactile"][name][field]["data"])
            for kind in ("empty", "grasp"):
                a, b = scene["references"].get(kind), references.get(kind)
                errors[f"reference.{kind}.{name}.{field}"] = array_error(a["images"][name][field], b["images"][name][field]) if a and b else {"compatible": False, "max_abs": None, "rmse": None}
    for name, pose in scene["training_info"].get("actor", {}).items():
        errors[f"offline_actor.{name}"] = array_error(pose, training_info.get("actor", {}).get(name))
    for name, value in scene["initial_prefix_state"].items():
        errors[f"initialization.{name}"] = array_error(value, initial_state[name])
    expected_history = scene["history"]
    history_match = len(history) == len(expected_history)
    time_match = task_elapsed == scene["task_elapsed_steps"]
    if history_match:
        for i, (left, right) in enumerate(zip(expected_history, history)):
            errors[f"history.{i}.joint_position"] = array_error(left["observation"]["proprioception"]["joint_position"], right["observation"]["proprioception"]["joint_position"])
            history_match &= digest(left["action"]) == digest(right["action"])
            for name in ("left_tactile", "right_tactile"):
                for field in ("rgb", "rgb_marker"):
                    a = left["observation"]["images"]["tactile"][name][field]
                    b = right["observation"]["images"]["tactile"][name][field]
                    history_match &= bool(a["valid"] and b["valid"])
                    errors[f"history.{i}.{name}.{field}"] = array_error(a["data"], b["data"])
    if task_start_physics is not None:
        time_match &= (observation["physics_step"]-task_start_physics == original["physics_step"]-scene["task_start_physics"])
        for left, right in zip(expected_history, history):
            time_match &= (left["observation"]["physics_step"]-scene["task_start_physics"] == right["observation"]["physics_step"]-task_start_physics)
    for group_name, group in original["images"].items():
        for name, sensor in group.items():
            for field, modality in sensor.items():
                other = observation["images"][group_name][name][field]
                time_match &= modality["age_steps"] == other["age_steps"]
    references_match = True
    for kind in ("empty", "grasp"):
        a, b = scene["references"].get(kind), references.get(kind)
        references_match &= bool(a and b and a["valid"] and b["valid"])
        if a and b and initial_prefix_physics is not None:
            time_match &= a["physics_step"]-scene["initial_prefix_physics"] == b["physics_step"]-initial_prefix_physics
    valid_sensors = all(v["valid"] for group in observation["images"].values() for sensor in group.values() for v in sensor.values())
    calibrated = tolerances is not None and tolerances.get("version") != "uncalibrated"
    covered = calibrated and all(key in tolerances["max_abs"] for key in errors)
    within = bool(covered and all(e["compatible"] and e["max_abs"] <= tolerances["max_abs"][key]
                                  and ("rmse" not in tolerances or e["rmse"] <= tolerances["rmse"].get(key, -1))
                                  for key, e in errors.items()))
    return {"parent_scene_id": scene["parent_scene_id"], "errors": errors,
            "tolerance_version": tolerances["version"] if calibrated else "uncalibrated",
            "status": "calibrated" if calibrated else "uncalibrated",
            "history_match": bool(history_match), "history_length": len(history),
            "history_mask": [False]*(8-min(8, len(history))) + [True]*min(8, len(history)),
            "time_match": bool(time_match), "valid_sensors": valid_sensors, "references_match": references_match,
            "valid_match": bool(within and history_match and time_match and valid_sensors and references_match)}


def calibrate_tolerances(reports, development_parents, version, margin=1.2, numeric_floor=0.0):
    if not np.isfinite(numeric_floor) or numeric_floor < 0:
        raise ValueError("Numeric precision floor must be finite and nonnegative")
    if not np.isfinite(margin) or margin < 1:
        raise ValueError("Calibration margin must be finite and at least one")
    if version == "uncalibrated" or len(development_parents) < 2 or len(reports) < 4:
        raise ValueError("Calibration requires repeated runs from at least two independent dev parents")
    if any(r["parent_scene_id"] not in development_parents or not r["history_match"] or not r["time_match"] or not r["valid_sensors"] or not r.get("references_match", False) for r in reports):
        raise ValueError("Invalid calibration cohort")
    from collections import Counter
    counts = Counter(r["parent_scene_id"] for r in reports)
    if set(counts) != set(development_parents) or any(n < 2 for n in counts.values()):
        raise ValueError("Each calibration parent requires at least two repetitions")
    keys = set(reports[0]["errors"])
    if any(set(r["errors"]) != keys or any(not e["compatible"] for e in r["errors"].values()) for r in reports):
        raise ValueError("Calibration metrics mismatch")
    def floor(key):
        # Image statistics use measured repeatability only. State comparisons
        # can declare a numerical floor separately from physical variability.
        return 0.0 if "tactile" in key or key.startswith("reference.") else numeric_floor
    return {"version": version, "statistic": "maximum_over_dev_repeats_times_margin_with_declared_numeric_floor",
            "margin": margin, "numeric_floor": numeric_floor,
            "development_parents": sorted(development_parents),
            **{metric: {k: max(max(r["errors"][k][metric] for r in reports)*margin, floor(k)) for k in keys}
               for metric in ("max_abs", "rmse")},
            "report_sha256": digest(reports)}
