"""Audit a finished FTP-1 rollout and export its actual rendered camera frames."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.scripts.audit_rollouts import audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    directory = output_path(Path("runs/phase1")/args.run_id)
    checks = json.loads((directory/"checks.json").read_text())
    episode = Path(checks["episode_path"])
    audit = audit_episode(episode)
    streams, joints = {}, []
    video = None
    try:
        with h5py.File(episode, "r") as handle:
            keys = sorted(handle["observations"], key=int)
            for key in keys:
                obs = read_tree(handle["observations"][key])
                joints.append(obs["proprioception"]["joint_position"])
                for category, sensors in obs["images"].items():
                    for sensor, modalities in sensors.items():
                        for modality, sample in modalities.items():
                            name = f"{category}_{sensor}_{modality}"
                            data = np.asarray(sample["data"])
                            row = streams.setdefault(name, {"frames": 0, "all_valid": True, "all_nonconstant": True,
                                                           "max_age_steps": 0, "shape": list(data.shape)})
                            row["frames"] += 1
                            row["all_valid"] &= bool(sample["valid"])
                            row["all_nonconstant"] &= bool(np.ptp(data.astype(float)) > 0)
                            row["max_age_steps"] = max(row["max_age_steps"], int(sample["age_steps"]))
                            if data.ndim == 3 and data.shape[-1] == 3 and data.dtype == np.uint8 and key in (keys[0], keys[-1]):
                                suffix = "first" if key == keys[0] else "last"
                                if not cv2.imwrite(str(directory/f"{name}_{suffix}.png"), cv2.cvtColor(data, cv2.COLOR_RGB2BGR)):
                                    raise RuntimeError("Failed to save rendering evidence")
                frame = np.asarray(obs["images"]["camera"]["head"]["rgb"]["data"])
                if video is None:
                    video = cv2.VideoWriter(str(directory/"head_camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                            (frame.shape[1], frame.shape[0]))
                    if not video.isOpened():
                        raise RuntimeError("Video writer failed")
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if video is not None:
            video.release()
    latencies = [json.loads(line)["wall_seconds"] for line in (directory/"policy/inferences.jsonl").read_text().splitlines()]
    joint_array = np.asarray(joints)
    report = {"audit": audit, "streams": streams, "controls": len(checks["controls"]),
              "all_controls_six_steps": all(c["physics_steps"] == 6 for c in checks["controls"]),
              "inference_seconds": {"first": latencies[0], "mean": float(np.mean(latencies)), "max": max(latencies)},
              "observed_joint_range_radians": np.ptp(joint_array, axis=0).tolist(),
              "task_success": checks["task_success"], "outcome": checks["outcome"],
              "video_fps": 20, "video_time_basis": "simulation time; inference wall time excluded"}
    (directory/"rendering_and_execution_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"audit": audit["status"], "controls": report["controls"], "streams": streams,
                      "inference_seconds": report["inference_seconds"], "task_success": checks["task_success"]}, indent=2))
    if audit["errors"] or not report["all_controls_six_steps"] or not all(s["all_valid"] and s["all_nonconstant"] for s in streams.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
