"""Align saved inference, accepted targets, tactile frames and failure metadata."""
import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.policy.ftp1_adapter import encode_observation


def official_mapping(source):
    """Load only pure mapping functions from the pinned source, without Isaac/model imports."""
    names = {"ActionMapping", "_canonicalize_action_joint_rep", "_infer_ftp1_layout",
             "_build_ftp1_state_from_univtac_qpos", "_extract_univtac_action_from_ftp1",
             "_resolve_univtac_abs_action_from_ftp1"}
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names):
        raise ValueError("Official source mapping definitions changed")
    scope = {"np": np, "dataclass": dataclass, "Any": Any,
             "FTP1_SINGLE_ARM_ACTION_REP_DIM": 48, "FTP1_RESERVED_ACTION_DIM": 15}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    return scope


def diagnose(run_id):
    directory = output_path(Path("runs/phase1")/run_id)
    checks = json.loads((directory/"checks.json").read_text())
    source = ROOT/".cache/sources/ftp1-policy/UniVTAC/policy/FTP1/deploy_policy.py"
    official = official_mapping(source)
    mapping = official["ActionMapping"]()
    rows, montage = [], []
    video = None
    config = checks["model"]["config"]
    first = config["chunk_first_index"]
    stride = config["chunk_stride"]
    previous_request, chunk_offset = None, 0
    try:
        with h5py.File(checks["episode_path"], "r") as handle:
            references = read_tree(handle["references"])
            selected = set(np.linspace(0, len(checks["controls"])-1, 6, dtype=int))
            for index, control in enumerate(checks["controls"]):
                obs = read_tree(handle["observations"][str(index)])
                after = read_tree(handle["observations"][str(index+1)])
                action = read_tree(handle["actions"][str(index)])
                training = read_tree(handle["training_info"][str(index)])
                request = control["inference"]["request_id"]
                if request != previous_request:
                    chunk_offset = 0
                    inference_obs = obs
                    previous_request = request
                else:
                    chunk_offset += 1
                inputs, base = encode_observation(inference_obs)
                state = official["_build_ftp1_state_from_univtac_qpos"](base, 120, mapping)
                with np.load(directory/f"policy/input_{request:05d}.npz") as saved:
                    input_error = max(float(np.max(np.abs(saved[k].astype(float)-v))) for k, v in inputs.items())
                    state_error = float(np.max(np.abs(saved["state"]-state)))
                chunk = np.load(directory/f"policy/chunk_{request:05d}.npy")
                chunk_index = first+chunk_offset*stride
                target = official["_resolve_univtac_abs_action_from_ftp1"](
                    official["_extract_univtac_action_from_ftp1"](chunk[chunk_index], 120, mapping),
                    base, checks["model"]["action_joint_rep"])
                cmd = action["commanded_action"]
                fingers = np.asarray(obs["proprioception"]["finger_position"])
                row = {"control": index, "physics_start": action["physics_start"], "physics_end": action["physics_end"],
                       "sim_seconds": index*6/120, "inference_id": request, "chunk_index": chunk_index,
                       "saved_input_max_error": input_error, "official_state_max_error": state_error,
                       "official_target_max_error": float(np.max(np.abs(target-control.get("model_action", action["proposed_action"])["values"]))),
                       "model_to_executed_target_max_difference": float(np.max(np.abs(target-action["proposed_action"]["values"]))),
                       "gripper_target_mode": control.get("gripper_target_mode", "predicted"),
                       "joint_clip_max": float(np.max(np.abs(cmd["joint_clip_delta"]))),
                       "gripper_clip": cmd["gripper_clip_delta"], "gripper_target_m": cmd["gripper_target"],
                       "finger_left_m": float(fingers[0]), "finger_right_m": float(fingers[1]),
                       "finger_after_m": float(np.mean(after["proprioception"]["finger_position"])),
                       "tracking_joint_max_rad": float(np.max(np.abs(np.asarray(cmd["joint_target"])-after["proprioception"]["joint_position"]))),
                       "ee_z_m": float(after["proprioception"]["ee_position_world"][2]),
                       "object_z_m": float(training["actor"]["prism"][2]),
                       "object_lost": training["object_lost"],
                       "inhand_bias_m": training["task_metadata"].get("inhand_bias"),
                       "insertion_rel_z_m": float(training["task_metadata"]["rel_pose"][2])}
                panels = []
                for name in ("head", "wrist"):
                    panels.append(cv2.resize(after["images"]["camera"][name]["rgb"]["data"], (320, 240)))
                for sensor in ("left_tactile", "right_tactile"):
                    tactile = after["images"]["tactile"][sensor]["rgb_marker"]["data"]
                    panels.append(cv2.resize(tactile, (320, 240)))
                    for kind in ("empty", "grasp"):
                        delta = tactile.astype(float)-references[kind]["images"][sensor]["rgb_marker"].astype(float)
                        row[f"{sensor}_{kind}_rmse"] = float(np.sqrt(np.mean(delta**2)))
                frame = np.concatenate(panels, axis=1)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"control={index} physics={action['physics_end']} grip={cmd['gripper_target']*1000:.3f} mm lost={training['object_lost']}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
                if video is None:
                    video = cv2.VideoWriter(str(directory/"synchronized_diagnosis.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 240))
                    if not video.isOpened():
                        raise RuntimeError("Could not create synchronized video")
                video.write(frame)
                if index in selected:
                    montage.append(frame)
                rows.append(row)
    finally:
        if video is not None:
            video.release()
    with (directory/"diagnosis_timeline.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(directory/"diagnosis_contact_sheet.jpg"), np.concatenate(montage))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for key in ("gripper_target_m", "finger_left_m", "finger_right_m"):
        axes[0].plot([r["control"] for r in rows], [r[key]*1000 for r in rows], label=key)
    axes[0].set_ylabel("single-finger position (mm)")
    for sensor in ("left_tactile", "right_tactile"):
        axes[1].plot([r[f"{sensor}_grasp_rmse"] for r in rows], label=sensor)
    axes[1].set_ylabel("RGB RMSE vs grasp reference")
    for key in ("ee_z_m", "object_z_m"):
        axes[2].plot([r[key]*1000 for r in rows], label=key)
    axes[2].set_ylabel("world height (mm)")
    axes[2].set_xlabel("control index (20 Hz simulated time)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(directory/"diagnosis_timeline.png")
    plt.close(fig)
    report = {"run_id": run_id, "controls": len(rows), "outcome": checks["outcome"],
              "official_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "max_errors": {k: max(r[k] for r in rows) for k in ("saved_input_max_error", "official_state_max_error", "official_target_max_error", "model_to_executed_target_max_difference", "joint_clip_max", "tracking_joint_max_rad")},
              "gripper_target_range_m": [min(r["gripper_target_m"] for r in rows), max(r["gripper_target_m"] for r in rows)],
              "initial_fingers_m": [rows[0]["finger_left_m"], rows[0]["finger_right_m"]],
              "final_inhand_bias_m": rows[-1]["inhand_bias_m"],
              "source_defaults": {"physics_hz": 120, "physics_steps_per_qpos_action": 1, "force": True, "chunk_len": 16},
              "current": {"physics_hz": 120, "physics_steps_per_qpos_action": 6, "force": False, "chunk_len": config["execute_chunk_steps"]},
              "attribution": "Mapping parity is measured; controller/cadence/chunk differences remain confounders. Tactile RGB change is not a calibrated force or contact label."}
    (directory/"diagnosis.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run_id), indent=2))


if __name__ == "__main__":
    main()
