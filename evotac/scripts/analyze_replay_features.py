"""Measure replay image/encoder discrepancies without changing frozen match gates."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from evotac.config import ROOT, output_path
from evotac.policy.tactile_features import FrozenTactileEncoder, SENSORS


def pose_discrepancy(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != (7,) or right.shape != (7,) or not np.isfinite([left, right]).all():
        raise ValueError("Expected finite xyz/wxyz poses")
    a, b = left[3:], right[3:]
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        raise ValueError("Invalid pose quaternion")
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    if a @ b < 0:
        b = -b
    # The norm form stays precise for small angles and treats q/-q equally.
    angle = 4*np.arctan2(np.linalg.norm(a-b), np.linalg.norm(a+b))
    return {"position_distance_m": float(np.linalg.norm(right[:3]-left[:3])),
            "position_delta_m": (right[:3]-left[:3]).tolist(), "rotation_distance_deg": float(np.rad2deg(angle))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-id", default="phase2_baseline_v1")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    directory = output_path(Path("runs/phase2")/args.run_id)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/"runs/phase2"/args.baseline_id/"manifest.json").read_text())
    provenance = json.loads((ROOT/"checkpoints/univtac_release/encoder_source.json").read_text())
    torch.set_num_threads(4)
    encoder = FrozenTactileEncoder(ROOT/"checkpoints/univtac_release/encoder.pth", provenance["sha256"])
    results = []
    for entry in manifest["rows"]:
        if entry["status"] != "finished":
            continue
        run = ROOT/"runs/phase1"/entry["run_id"]
        checks = json.loads((run/"checks.json").read_text())
        if "replay_episode_path" not in checks:
            continue
        frames, images = [], [[], []]
        with h5py.File(checks["episode_path"], "r") as source, h5py.File(checks["replay_episode_path"], "r") as replay:
            count = len(replay["observations"])
            for index in range(count):
                a, b = source[f"observations/{index}"], replay[f"observations/{index}"]
                row = {"control_boundary": index, "physics_source": int(a["physics_step"][()]),
                       "physics_replay": int(b["physics_step"][()]), "image_errors": {}}
                row["prism_pose_error"] = (pose_discrepancy(source[f"training_info/{index-1}/actor/prism"][()],
                                                            replay[f"training_info/{index-1}/actor/prism"][()]) if index else None)
                for category, sensors, field in (("camera", ("head", "wrist"), "rgb"),
                                                  ("tactile", SENSORS, "rgb_marker")):
                    for sensor in sensors:
                        key = f"images/{category}/{sensor}/{field}/data"
                        left, right = a[key][()], b[key][()]
                        delta = left.astype(float)-right.astype(float)
                        row["image_errors"][f"{category}/{sensor}/{field}"] = {
                            "max_abs": float(np.abs(delta).max()), "rmse": float(np.sqrt((delta**2).mean()))}
                        if category == "tactile":
                            images[0].append(left)
                            images[1].append(right)
                frames.append(row)
        embeddings = []
        for side in images:
            values = [encoder(np.stack(side[i:i+8])) for i in range(0, len(side), 8)]
            embeddings.append(np.concatenate(values).reshape(count, 2, 512))
        source_features, replay_features = embeddings
        distance = np.linalg.norm(source_features-replay_features, axis=-1)
        relative = distance/np.maximum(np.linalg.norm(source_features, axis=-1), 1e-12)
        temporal = np.linalg.norm(np.diff(source_features, axis=0), axis=-1)
        for index, row in enumerate(frames):
            row["encoder_l2_difference"] = distance[index].tolist()
            row["encoder_relative_l2_difference"] = relative[index].tolist()
            row["source_adjacent_feature_l2"] = temporal[index-1].tolist() if index else None
        result = {"seed": entry["seed"], "run_id": entry["run_id"], "frozen_v3_match": checks["reconstruction"]["valid_match"],
                  "encoder_relative_l2_mean": float(relative.mean()), "encoder_relative_l2_max": float(relative.max()),
                  "encoder_l2_mean": float(distance.mean()), "source_adjacent_feature_l2_mean": float(temporal.mean()),
                  "frames": frames}
        results.append(result)
        np.savez_compressed(directory/f"seed{entry['seed']}_features.npz", source=source_features, replay=replay_features)
        print(json.dumps({k: v for k, v in result.items() if k != "frames"}), flush=True)
    report = {"baseline_id": args.baseline_id, "encoder": encoder.version, "scenes": results,
              "scope": "Diagnostic only: encoder closeness does not establish equivalent contacts or outcome effects. Frozen v3 decisions are unchanged.",
              "camera_scope": "Camera pixel metrics are additional diagnostics; v3 only gated their freshness, not pixel errors."}
    (directory/"report.json").write_text(json.dumps(report, indent=2))
    if not results:
        raise ValueError("No completed replay pairs available")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row in results:
        axes[0].plot([max(f["image_errors"][f"tactile/{s}/rgb_marker"]["rmse"] for s in SENSORS) for f in row["frames"]], label=str(row["seed"]))
        axes[1].plot([max(f["encoder_relative_l2_difference"]) for f in row["frames"]], label=str(row["seed"]))
    axes[0].set_ylabel("max over fingers: image RMSE (0-255)")
    axes[1].set_ylabel("max over fingers: relative feature L2")
    for ax in axes:
        ax.set_xlabel("control boundary")
        ax.grid(alpha=.25)
        ax.legend(title="seed")
    fig.tight_layout()
    fig.savefig(directory/"replay_discrepancies.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
