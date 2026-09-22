"""Read-only snapshot diagnostics, never a replacement for the causal gate."""
import hashlib

import numpy as np
import torch


def _diagnostic_array(value):
    """Return a compact, JSON-safe summary and a CPU copy for comparison."""
    if value is None:
        return None, None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": False}, None
    array = np.ascontiguousarray(array)
    payload = array.view(np.uint8).tobytes()
    return {
        "shape": list(array.shape), "dtype": str(array.dtype), "finite": True,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "mean": float(array.mean()) if array.size else 0.0,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, array.copy()


def _sensor_diagnostics(wrapper):
    """Capture raw sensor buffers without serializing image-sized arrays."""
    task = wrapper.task
    public = {"uipc_frame": int(task.uipc_sim.world.frame()),
              "physics_step": int(task._physics_step_count), "sensors": {}}
    raw = {}
    objects = [("camera", name, sensor) for name, sensor in task._camera_manager.cameras.items()]
    objects += [("tactile", name, tactile.sensor)
                for name, tactile in task._tactile_manager.tactiles.items()]
    for category, name, sensor in objects:
        key = f"{category}.{name}"
        row = {"frame": _json_scalar(getattr(sensor, "_frame", None)),
               "timestamp": _json_scalar(getattr(sensor, "_timestamp", None)),
               "buffers": {}}
        raw[key] = {}
        output = getattr(getattr(sensor, "_data", None), "output", {}) or {}
        output = dict(output)
        if category == "tactile":
            output["indentation_depth"] = getattr(sensor, "_indentation_depth", None)
            camera = getattr(sensor, "camera", None)
            camera_data = getattr(camera, "_data", None)
            for field, value in (getattr(camera_data, "output", {}) or {}).items():
                output[f"camera_{field}"] = value
            row["camera_frame"] = _json_scalar(getattr(camera, "_frame", None))
            row["camera_timestamp"] = _json_scalar(getattr(camera, "_timestamp", None))
        fields = ("rgb", "height_map", "marker_motion", "tactile_rgb", "marker_rgb",
                  "indentation_depth", "camera_depth")
        for field in fields:
            if field not in output:
                continue
            summary, array = _diagnostic_array(output[field])
            row["buffers"][field] = summary
            if array is not None:
                raw[key][field] = array
        public["sensors"][key] = row
    # Query retrieved UIPC geometry directly, rather than a potentially stale
    # IsaacLab lazy data buffer. No physics or renderer update is performed.
    geometry = {"surface_positions": task.uipc_sim.sio.simplicial_surface(2).positions().view()}
    for name, tactile in task._tactile_manager.tactiles.items():
        slot, _ = task.uipc_sim.scene.geometries().find(tactile.gelpad.obj_id)
        geometry[f"{name}_positions"] = slot.geometry().positions().view()
    raw["uipc"] = {}
    public["sensors"]["uipc"] = {"buffers": {}}
    for field, value in geometry.items():
        summary, array = _diagnostic_array(value)
        public["sensors"]["uipc"]["buffers"][field] = summary
        if array is not None:
            raw["uipc"][field] = array
    return public, raw


def _json_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value) if value is not None else None
    if array is None:
        return None
    return array.tolist() if array.ndim else array.item()


def _compare_sensor_diagnostics(source, candidate):
    same_step = (source[0]["physics_step"] == candidate[0]["physics_step"]
                 and source[0]["uipc_frame"] == candidate[0]["uipc_frame"])
    if not same_step:
        raise ValueError("sensor diagnostics must compare the same physics step and UIPC frame")
    differences = {}
    for key in source[1].keys() | candidate[1].keys():
        source_fields = source[1].get(key, {})
        candidate_fields = candidate[1].get(key, {})
        for field in source_fields.keys() | candidate_fields.keys():
            left = source_fields.get(field)
            right = candidate_fields.get(field)
            if left is None or right is None or left.shape != right.shape:
                differences[f"{key}.{field}"] = {"compatible": False, "max_abs": None}
            else:
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                differences[f"{key}.{field}"] = {
                    "compatible": True,
                    "max_abs": float(delta.max()) if delta.size else 0.0,
                    "rmse": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
                    "source_sha256": hashlib.sha256(np.ascontiguousarray(left).view(np.uint8).tobytes()).hexdigest(),
                    "candidate_sha256": hashlib.sha256(np.ascontiguousarray(right).view(np.uint8).tobytes()).hexdigest(),
                }
    return {"same_physics_step": same_step, "source": source[0], "candidate": candidate[0], "differences": differences}
