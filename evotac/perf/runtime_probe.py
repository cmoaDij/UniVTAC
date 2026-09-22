"""Small, dependency-free probes for wall-clock simulation performance."""
from __future__ import annotations

import csv
import io
import os
import subprocess
import time
from collections.abc import Iterable


def gpu_snapshot():
    """Return a best-effort nvidia-smi snapshot, or an explicit unavailable record."""
    query = "index,name,utilization.gpu,memory.used,memory.total,uuid"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 6:
            continue
        try:
            utilization = row[2].strip()
            # N/A can indicate an unhealthy device; retain it for reporting
            # but do not approve that device for CUDA work.
            utilization_value = (None if utilization in {"[N/A]", "N/A"}
                                 else float(utilization))
            rows.append({"index": int(row[0].strip()), "name": row[1].strip(),
                         "uuid": row[5].strip(),
                         "utilization_gpu_percent": utilization_value,
                         "memory_used_mib": int(float(row[3].strip())),
                         "memory_total_mib": int(float(row[4].strip()))})
        except ValueError:
            continue
    return {"available": bool(rows), "captured_at_unix": time.time(), "gpus": rows}


def gpu_preflight(snapshot, *, device="cuda:0", min_free_mib=20000, visible_devices=None):
    """Check that a requested CUDA device has enough uncommitted memory.

    This is a launch guard for profiling only. It cannot predict all Isaac or
    model allocations, but it avoids starting a long reset when a training job
    has already consumed the device. ``device='cpu'`` is always accepted.
    """
    if str(device).lower() == "cpu":
        return {"ok": True, "reason": "cpu_device"}
    text = str(device).lower()
    try:
        logical_index = int(text.split(":", 1)[1]) if ":" in text else 0
    except ValueError as exc:
        raise ValueError(f"invalid CUDA device: {device}") from exc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if visible and visible.strip() not in {"", "-1"}:
        mapping = [item.strip() for item in visible.split(",") if item.strip()]
        if logical_index >= len(mapping):
            return {"ok": False, "reason": "logical_device_not_mapped", "device": logical_index,
                    "visible_devices": visible}
        selected = mapping[logical_index]
        if selected.startswith("GPU-"):
            candidates = [row for row in snapshot.get("gpus", []) if row.get("uuid", "").startswith(selected)]
            if len(candidates) != 1:
                return {"ok": False, "reason": "uuid_not_uniquely_mapped", "uuid": selected}
            index = candidates[0]["index"]
        elif selected.isdigit():
            index = int(selected)
        else:
            return {"ok": False, "reason": "logical_device_not_mapped", "visible_devices": visible}
    else:
        index = logical_index
    if type(min_free_mib) is not int or min_free_mib < 0:
        raise ValueError("min_free_mib must be a nonnegative integer")
    if not snapshot.get("available"):
        return {"ok": False, "reason": "nvidia_smi_unavailable", "device": index}
    rows = [row for row in snapshot["gpus"] if row["index"] == index]
    if not rows:
        return {"ok": False, "reason": "device_not_found", "device": index}
    row = rows[0]
    if row["utilization_gpu_percent"] is None:
        return {"ok": False, "reason": "gpu_telemetry_unavailable", "device": index}
    free = int(row["memory_total_mib"] - row["memory_used_mib"])
    return {"ok": free >= min_free_mib, "reason": "enough_free_memory" if free >= min_free_mib else "insufficient_free_memory",
            "device": index, "free_mib": free, "required_free_mib": min_free_mib,
            "utilization_gpu_percent": row["utilization_gpu_percent"]}


def summarize_rates(executions: Iterable[dict]):
    """Compute actual wall-clock rates from fixed-step execution records."""
    rows = [row for row in executions if row.get("status") == "executed" and row.get("physics_steps", 0) > 0]
    physics_steps = sum(int(row["physics_steps"]) for row in rows)
    wall_seconds = sum(float(row.get("wall_seconds", 0.0)) for row in rows)
    controls = len(rows)
    if wall_seconds <= 0:
        return {"controls": controls, "physics_steps": physics_steps, "wall_seconds": wall_seconds,
                "wall_control_hz": None, "wall_physics_hz": None}
    return {"controls": controls, "physics_steps": physics_steps, "wall_seconds": wall_seconds,
            "wall_control_hz": controls / wall_seconds, "wall_physics_hz": physics_steps / wall_seconds}
