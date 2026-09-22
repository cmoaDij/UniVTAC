"""Produce staged source/replay diagnostics for frozen FTP-1 chunk16 prefixes.

The report deliberately leaves ``replay_tolerances_v3.json`` untouched.  It is a
diagnostic view of where a prefix first diverges, rather than a new acceptance
gate or a way to turn an unmatched prefix into a training sample.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree


BOUNDARIES = (0, 1, 4, 8, 12)
IMAGE_FIELDS = (
    ("camera", "head", "rgb"),
    ("camera", "wrist", "rgb"),
    ("tactile", "left_tactile", "rgb"),
    ("tactile", "left_tactile", "rgb_marker"),
    ("tactile", "right_tactile", "rgb"),
    ("tactile", "right_tactile", "rgb_marker"),
)


def _array(a):
    return np.asarray(a, dtype=np.float64)


def _error(a, b):
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        return {"compatible": False, "max_abs": None, "rmse": None}
    delta = a - b
    return {"compatible": True, "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0}


def _pose_error(a, b):
    a, b = _array(a), _array(b)
    result = _error(a, b)
    if a.shape == (7,) and b.shape == (7,):
        qa, qb = a[3:], b[3:]
        qa /= max(np.linalg.norm(qa), 1e-12)
        qb /= max(np.linalg.norm(qb), 1e-12)
        if np.dot(qa, qb) < 0:
            qb = -qb
        result["position_distance_m"] = float(np.linalg.norm(a[:3] - b[:3]))
        result["rotation_distance_deg"] = float(np.rad2deg(4 * np.arctan2(
            np.linalg.norm(qa - qb), np.linalg.norm(qa + qb))))
    return result


def _max_image_error(source_obs, replay_obs):
    values = {}
    for category, sensor, field in IMAGE_FIELDS:
        left = source_obs["images"][category][sensor][field]["data"]
        right = replay_obs["images"][category][sensor][field]["data"]
        values[f"{category}.{sensor}.{field}"] = _error(left, right)
    return values


def _references(source, replay):
    result = {}
    for kind in ("empty", "grasp"):
        for category, sensor, field in IMAGE_FIELDS:
            # References contain tactile images only; retaining the same field
            # names in the output makes the stage report easy to query.
            if category != "tactile":
                continue
            left = source["references"][kind]["images"][sensor][field]
            right = replay["references"][kind]["images"][sensor][field]
            result[f"{kind}.{sensor}.{field}"] = _error(left, right)
    return result


def _actor_pose(handle, index):
    if index == 0 or str(index - 1) not in handle["training_info"]:
        return None
    info = read_tree(handle["training_info"][str(index - 1)])
    return info.get("actor", {}).get("prism")


def _boundary_row(source, replay, index):
    source_obs = read_tree(source["observations"][str(index)])
    replay_obs = read_tree(replay["observations"][str(index)])
    row = {
        "boundary": index,
        "stage": "approach_end" if index == 0 else f"policy_after_{index}",
        "source_physics_step": int(source_obs["physics_step"]),
        "replay_physics_step": int(replay_obs["physics_step"]),
        "physics_delta": int(replay_obs["physics_step"]) - int(source_obs["physics_step"]),
        "joint_position": _error(source_obs["proprioception"]["joint_position"], replay_obs["proprioception"]["joint_position"]),
        "finger_position": _error(source_obs["proprioception"]["finger_position"], replay_obs["proprioception"]["finger_position"]),
        "ee_position_world": _error(source_obs["proprioception"]["ee_position_world"], replay_obs["proprioception"]["ee_position_world"]),
        "ee_quaternion_world": _error(source_obs["proprioception"]["ee_quaternion_world"], replay_obs["proprioception"]["ee_quaternion_world"]),
        "images": _max_image_error(source_obs, replay_obs),
    }
    source_actor, replay_actor = _actor_pose(source, index), _actor_pose(replay, index)
    row["object_pose"] = None if source_actor is None or replay_actor is None else _pose_error(source_actor, replay_actor)
    return row


def _prefix_report(source, replay):
    def reset_event(handle):
        for key in handle["events"]:
            event = read_tree(handle["events"][key])
            if event.get("kind") == "reset_complete":
                return event
        return None

    left, right = reset_event(source), reset_event(replay)
    if left is None or right is None:
        return {"available": False}
    lp, rp = left.get("prefix", []), right.get("prefix", [])
    hold_left = sum(int(e.get("physics_steps", 0)) for e in lp if e.get("kind") == "hold")
    hold_right = sum(int(e.get("physics_steps", 0)) for e in rp if e.get("kind") == "hold")
    return {
        "available": True,
        "event_count_source": len(lp), "event_count_replay": len(rp),
        "event_count_delta": len(rp) - len(lp),
        "hold_physics_source": hold_left, "hold_physics_replay": hold_right,
        "hold_physics_delta": hold_right - hold_left,
        "initial_prefix_physics_source": left.get("decision_start_physics"),
        "initial_prefix_physics_replay": right.get("decision_start_physics"),
        "prefix_event_sequence_equal": [e.get("kind") for e in lp] == [e.get("kind") for e in rp],
    }


def _action_report(source, replay, count):
    rows = []
    for index in range(count):
        left, right = read_tree(source["actions"][str(index)]), read_tree(replay["actions"][str(index)])
        lc, rc = left["commanded_action"], right["commanded_action"]
        arm = _error(lc["joint_target"], rc["joint_target"])
        grip = _error([lc["gripper_target"]], [rc["gripper_target"]])
        rows.append({"control": index, "joint_target": arm, "gripper_target": grip,
                     "source_status": left.get("status"), "replay_status": right.get("status")})
    return rows


def _first_divergence(boundaries):
    out = {}
    for category, getter in {
        "joint_position": lambda r: r["joint_position"]["max_abs"],
        "finger_position": lambda r: r["finger_position"]["max_abs"],
        "ee_position_world": lambda r: r["ee_position_world"]["max_abs"],
        "object_position_m": lambda r: (r["object_pose"] or {}).get("position_distance_m", 0.0),
        "object_rotation_deg": lambda r: (r["object_pose"] or {}).get("rotation_distance_deg", 0.0),
        "tactile_rgb_marker_max_abs": lambda r: max(v["max_abs"] for k, v in r["images"].items() if "tactile" in k and "rgb_marker" in k),
    }.items():
        # These are diagnostic floors only; they do not replace v3 gates.
        floor = 1e-7 if category not in {"object_rotation_deg", "tactile_rgb_marker_max_abs"} else (1e-4 if category == "object_rotation_deg" else 0.0)
        match = next((r for r in boundaries if getter(r) > floor), None)
        out[category] = None if match is None else {"boundary": match["boundary"], "stage": match["stage"], "value": getter(match)}
    return out


def diagnose(run_id, output_dir=None):
    root = output_path(Path("runs/phase1") / run_id)
    checks = json.loads((root / "checks.json").read_text())
    if not checks.get("replay_episode_path"):
        raise ValueError(f"{run_id} has no replay episode")
    source_value = checks.get("episode_path") or checks.get("source_episode_path")
    if not source_value:
        raise ValueError(f"{run_id} has no source episode path")
    source_path, replay_path = Path(source_value), Path(checks["replay_episode_path"])
    with h5py.File(source_path, "r") as source, h5py.File(replay_path, "r") as replay:
        count = min(len(source["observations"]), len(replay["observations"]))
        if count < 13:
            raise ValueError(f"{run_id} has only {count} paired observations; expected 13")
        boundaries = [_boundary_row(source, replay, i) for i in BOUNDARIES]
        report = {
            "run_id": run_id, "seed": checks.get("seed"), "policy_config": "configs/ftp1_insert_hole_chunk16.yaml",
            "replay_tolerance_version": checks.get("reconstruction", {}).get("tolerance_version"),
            "frozen_v3_match": checks.get("reconstruction", {}).get("valid_match"),
            "hardware_comparison": checks.get("hardware_comparison"),
            "baseline_outcome": checks.get("outcome"),
            "reconstruction_flags": {k: checks.get("reconstruction", {}).get(k) for k in ("valid_match", "history_match", "time_match", "valid_sensors", "references_match")},
            "references": _references(source, replay), "prefix_execution": _prefix_report(source, replay),
            "boundaries": boundaries, "first_divergence": _first_divergence(boundaries),
            "actions": _action_report(source, replay, min(12, len(source["actions"]), len(replay["actions"]))),
            "diagnostic_thresholds": {"state_max_abs": 1e-7, "object_rotation_deg": 1e-4, "tactile_max_abs": 0.0},
            "interpretation": "Diagnostic metrics only; v3 frozen acceptance gates are unchanged and unmatched prefixes are excluded from paired recovery labels.",
        }
    out = output_path(Path(output_dir) if output_dir else Path("runs/phase2") / f"{run_id}_diagnosis")
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with (out / "boundaries.csv").open("w", newline="") as handle:
        fields = ["boundary", "stage", "source_physics_step", "replay_physics_step", "physics_delta",
                  "joint_max_abs", "finger_max_abs", "ee_max_abs", "object_position_m", "object_rotation_deg",
                  "left_tactile_rgb_marker_max_abs", "right_tactile_rgb_marker_max_abs"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in report["boundaries"]:
            writer.writerow({"boundary": row["boundary"], "stage": row["stage"], "source_physics_step": row["source_physics_step"],
                             "replay_physics_step": row["replay_physics_step"], "physics_delta": row["physics_delta"],
                             "joint_max_abs": row["joint_position"]["max_abs"], "finger_max_abs": row["finger_position"]["max_abs"],
                             "ee_max_abs": row["ee_position_world"]["max_abs"],
                             "object_position_m": (row["object_pose"] or {}).get("position_distance_m"),
                             "object_rotation_deg": (row["object_pose"] or {}).get("rotation_distance_deg"),
                             "left_tactile_rgb_marker_max_abs": row["images"]["tactile.left_tactile.rgb_marker"]["max_abs"],
                             "right_tactile_rgb_marker_max_abs": row["images"]["tactile.right_tactile.rgb_marker"]["max_abs"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/phase2")
    args = parser.parse_args()
    summaries = []
    for run_id in args.run_id:
        report = diagnose(run_id, args.output_root / f"{run_id}_diagnosis")
        summaries.append({"run_id": run_id, "seed": report["seed"], "v3_match": report["frozen_v3_match"],
                          "first_divergence": report["first_divergence"]})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
