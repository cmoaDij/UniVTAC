"""Compare preselected development scenes, retaining every excluded attempt."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.scripts.audit_rollouts import audit_episode


def read_run(run_id):
    directory = ROOT/"runs/phase1"/run_id
    path = directory/"checks.json"
    if not path.exists():
        return {"run_id": run_id, "status": "not_available"}
    checks = json.loads(path.read_text())
    if checks.get("status") not in {"executed", "executed_partial"} or "episode_path" not in checks:
        return {"run_id": run_id, "status": checks.get("status"), "error": checks.get("error")}
    # Call only after the cohort supervisor has finalized this child. Reading
    # the source episode is safe even when its separate replay is being saved.
    with h5py.File(checks["episode_path"], "r") as handle:
        initial = read_tree(handle["observations/0"])
        metadata = json.loads(handle.attrs["scene_metadata"])
        finalized = bool(handle.attrs["complete"])
    manifest = json.loads((ROOT/"datasets/phase1"/run_id/"manifest.json").read_text())
    inference_path = directory/"policy/inferences.jsonl"
    latencies = [json.loads(line)["wall_seconds"] for line in inference_path.read_text().splitlines()]
    outcome = checks["outcome"]
    valid = finalized and outcome["valid_trial"] and not outcome["incomplete"]
    return {"run_id": run_id, "status": "finished", "valid_trial": bool(valid),
            "seed": checks["seed"], "parent_scene_id": metadata["parent_scene_id"], "condition": metadata["condition"],
            "success": checks["task_success"], "object_lost": outcome["reason"] == "object_lost", "reason": outcome["reason"],
            "controls": len(checks["controls"]), "physics_steps": sum(c["physics_steps"] for c in checks["controls"]),
            "inference_seconds": latencies,
            "latency": {"count": len(latencies), "first_s": latencies[0], "mean_s": float(np.mean(latencies)),
                        "p95_s": float(np.percentile(latencies, 95))},
            "initial_qpos": initial["proprioception"]["joint_position"].tolist(),
            "initial_fingers": initial["proprioception"]["finger_position"].tolist(),
            "initial_ee": initial["proprioception"]["ee_position_world"].tolist(),
            "policy_config": checks["model"]["config"], "config": manifest["config"],
            "source_hashes": manifest["evotac_file_hashes"], "episode_path": checks["episode_path"],
            "replay_match": checks.get("reconstruction", {}).get("valid_match"),
            "audit": audit_episode(Path(checks["episode_path"]))}


def pair(left, right):
    if left.get("status") != "finished" or right.get("status") != "finished":
        return {"status": "incomplete", "baseline": left, "variant": right}
    assert left["seed"] == right["seed"] and left["parent_scene_id"] == right["parent_scene_id"]
    assert left["condition"] == right["condition"]
    for key in ("simulation", "controller", "observation", "budgets", "replay"):
        assert left["config"][key] == right["config"][key], f"Changed physical contract: {key}"
    for key in ("source_revision", "checkpoint", "domain", "num_inference_steps", "inference_seed", "prompt"):
        assert left["policy_config"][key] == right["policy_config"][key], f"Policy identity differs: {key}"
    return {"status": "paired", "seed": left["seed"], "baseline": left, "variant": right,
            "initial_joint_max_difference_rad": float(np.max(np.abs(np.asarray(left["initial_qpos"])-right["initial_qpos"]))),
            "initial_finger_max_difference_m": float(np.max(np.abs(np.asarray(left["initial_fingers"])-right["initial_fingers"]))),
            "initial_ee_distance_m": float(np.linalg.norm(np.asarray(left["initial_ee"])-right["initial_ee"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-id", default="phase2_baseline_v1")
    parser.add_argument("--followups-id", default="phase2_followups_v3")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    directory = output_path(Path("runs/phase2")/args.run_id)
    directory.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((ROOT/"runs/phase2"/args.baseline_id/"manifest.json").read_text())
    selected, excluded = [], []
    for seed in baseline["scenes"]["seeds"]:
        row = read_run(f"{args.baseline_id}_seed{seed}")
        if seed == 24:
            # Predeclared provenance repeat replaces this scene regardless of
            # whether its outcome improves; the original remains in the report.
            excluded.append({"reason": "launch/import source hash ambiguity", "attempt": row})
            row = read_run(f"{args.followups_id}_seed24_provenance_repeat")
        selected.append(row)
    chunk = [pair(row, read_run(f"{args.followups_id}_chunk16_seed{seed}"))
             for row, seed in zip(selected, baseline["scenes"]["seeds"])]
    grip = pair(selected[0], read_run(f"{args.followups_id}_hold_gripper_seed0"))
    for row in selected:
        if row.get("status") == "finished":
            assert row["policy_config"]["execute_chunk_steps"] == 1
            assert row["policy_config"].get("gripper_target_mode", "predicted") == "predicted"
    for comparison in chunk:
        row = comparison["variant"]
        if row.get("status") == "finished":
            assert row["policy_config"]["execute_chunk_steps"] == 16
            assert row["policy_config"].get("gripper_target_mode", "predicted") == "predicted"
    if grip["variant"].get("status") == "finished":
        assert grip["variant"]["policy_config"]["execute_chunk_steps"] == 1
        assert grip["variant"]["policy_config"]["gripper_target_mode"] == "hold_initial"
    def summary(rows):
        completed = [r for r in rows if r.get("status") == "finished"]
        valid = [r for r in completed if r["valid_trial"]]
        latencies = [v for r in completed for v in r["inference_seconds"]]
        warm_latencies = [v for r in completed for v in r["inference_seconds"][1:]]
        return {"expected": len(rows), "finished": len(completed), "valid_trials": len(valid),
                "successes": sum(r["success"] for r in valid), "object_lost": sum(r["object_lost"] for r in valid),
                "outcome_counts": {reason: sum(r["reason"] == reason for r in valid)
                                   for reason in sorted({r["reason"] for r in valid})},
                "success_rate": sum(r["success"] for r in valid)/len(valid) if valid else None,
                "object_lost_rate": sum(r["object_lost"] for r in valid)/len(valid) if valid else None,
                "mean_controls": float(np.mean([r["controls"] for r in valid])) if valid else None,
                "inference_seconds": {"n": len(latencies), "mean": float(np.mean(latencies)),
                                      "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))} if latencies else None,
                "inference_seconds_excluding_first_per_scene": {
                    "n": len(warm_latencies), "mean": float(np.mean(warm_latencies)),
                    "p50": float(np.median(warm_latencies)), "p95": float(np.percentile(warm_latencies, 95))
                } if warm_latencies else None,
                "replay_attempts": sum(r["replay_match"] is not None for r in completed),
                "replay_matches": sum(r["replay_match"] is True for r in completed)}
    report = {"baseline": summary(selected), "chunk16": summary([p["variant"] for p in chunk]),
              "chunk_pairs": chunk, "gripper_seed0_pair": grip, "excluded_retained_attempts": excluded,
              "limitations": ["Development cohort only; no independent test performance claim",
                              "Separate simulator resets are seed-paired, not identical hidden-state counterfactuals",
                              "Initial-state differences are reported for every complete pair",
                              "Gripper intervention covers one scene and cannot establish a ten-scene recovery rate"]}
    report["status"] = "complete" if all(p["status"] == "paired" for p in chunk) and grip["status"] == "paired" else "incomplete"
    (directory/"comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("status", "baseline", "chunk16")}, indent=2))


if __name__ == "__main__":
    main()
