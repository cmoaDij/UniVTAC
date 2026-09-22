"""Frozen development cohort, fresh Isaac process per scene, durable denominators."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evotac.config import ROOT, output_path


def summarize(rows, expected):
    finished = [r for r in rows if r.get("status") == "finished"]
    valid = [r for r in finished if r.get("outcome", {}).get("valid_trial")
             and not r["outcome"].get("incomplete", True)]
    latencies = [v for r in finished for v in r.get("inference_seconds", [])]
    replay = [r["reconstruction"] for r in finished if "reconstruction" in r]
    pending = sum(r.get("status") == "running" for r in rows)
    return {"expected_scenes": expected, "attempted": len(rows), "finished": len(finished),
            "running": pending, "valid_trials": len(valid), "invalid_or_incomplete": len(rows)-pending-len(valid),
            "successes": sum(r.get("task_success", False) for r in valid),
            "success_rate_valid": sum(r.get("task_success", False) for r in valid)/len(valid) if valid else None,
            "object_lost_rate_valid": sum(r["outcome"]["reason"] == "object_lost" for r in valid)/len(valid) if valid else None,
            "executed_controls": [r.get("controls") for r in finished],
            "executed_controls_mean": float(np.mean([r["controls"] for r in finished])) if finished else None,
            "inference_seconds": {"n": len(latencies), "mean": float(np.mean(latencies)),
                                  "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))} if latencies else None,
            "replay_attempted": len(replay), "replay_matches": sum(r["valid_match"] for r in replay),
            "replay_match_rate": sum(r["valid_match"] for r in replay)/len(replay) if replay else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenes", type=Path, default=ROOT/"configs/phase2_dev_scenes.json")
    parser.add_argument("--policy-config", type=Path, default=ROOT/"configs/ftp1_insert_hole.yaml")
    parser.add_argument("--sim-gpu", default="0")
    parser.add_argument("--model-gpu", default="9")
    parser.add_argument("--replay-at", type=int, default=12)
    args = parser.parse_args()
    scenes = json.loads(args.scenes.read_text())
    directory = output_path(Path("runs/phase2")/args.run_id)
    directory.mkdir(parents=True, exist_ok=False)
    policy = directory/"policy.yaml"
    policy.write_bytes(args.policy_config.read_bytes())
    rows = []
    manifest = {"scenes": scenes, "policy_config_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "command": sys.argv, "started_unix": time.time(), "status": "running", "rows": rows}
    def save():
        manifest["summary"] = summarize(rows, len(scenes["seeds"]))
        temporary = directory/"manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(directory/"manifest.json")
    save()
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", CUDA_VISIBLE_DEVICES=args.sim_gpu,
               LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parent.parent/"lib")+":"+os.environ.get("LD_LIBRARY_PATH", ""))
    for seed in scenes["seeds"]:
        run_id = f"{args.run_id}_seed{seed}"
        command = [sys.executable, "-u", "-m", "evotac.scripts.run_ftp1", "--livestream", "2",
                   "--device", "cuda:0", "--seed", str(seed), "--run-id", run_id,
                   "--model-gpu", args.model_gpu, "--policy-config", str(policy),
                   "--max-controls", str(scenes["max_controls"]), "--replay-at", str(args.replay_at)]
        row = {"seed": seed, "run_id": run_id, "status": "running", "command": command}
        rows.append(row)
        save()
        with (directory/f"seed{seed}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            row["pid"] = process.pid
            save()
            row["exit_code"] = process.wait()
        checks_path = ROOT/"runs/phase1"/run_id/"checks.json"
        checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
        row.update(status="finished" if checks.get("status") in {"executed", "executed_partial"} else "failed",
                   outcome=checks.get("outcome", {}), task_success=checks.get("task_success", False),
                   controls=len(checks.get("controls", [])), episode_path=checks.get("episode_path"),
                   error=checks.get("error"))
        if "reconstruction" in checks:
            row["reconstruction"] = checks["reconstruction"]
        row["physics_steps"] = sum(c["physics_steps"] for c in checks.get("controls", []))
        row["simulated_seconds"] = row["physics_steps"]/120
        row["inference_first_seconds"] = None
        inference_path = checks_path.parent/"policy/inferences.jsonl"
        row["inference_seconds"] = [json.loads(line)["wall_seconds"] for line in inference_path.read_text().splitlines()] if inference_path.exists() else []
        if row["inference_seconds"]:
            row["inference_first_seconds"] = row["inference_seconds"][0]
        save()
        print(json.dumps({"seed": seed, "status": row["status"], "outcome": row["outcome"], "summary": manifest["summary"]}), flush=True)
        if row["status"] == "failed" or row["exit_code"] != 0:
            manifest["status"] = "stopped_infrastructure_error"
            save()
            raise SystemExit(1)
        report = subprocess.run([sys.executable, "-m", "evotac.scripts.report_ftp1_run", "--run-id", run_id],
                                stdout=subprocess.DEVNULL)
        row["report_exit_code"] = report.returncode
        save()
        if report.returncode:
            manifest["status"] = "stopped_audit_error"
            save()
            raise SystemExit(1)
    manifest["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
