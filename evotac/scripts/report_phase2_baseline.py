"""Audit finished attempts and report cohort coverage without dropping failures."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evotac.config import ROOT, output_path
from evotac.scripts.audit_rollouts import audit_episode
from evotac.scripts.run_phase2_baseline import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    directory = output_path(Path("runs/phase2")/args.run_id)
    manifest = json.loads((directory/"manifest.json").read_text())
    tolerances = json.loads((ROOT/manifest["scenes"]["replay_tolerances"]).read_text())
    rows = []
    failures = []
    for entry in manifest["rows"]:
        row = dict(entry)
        if row["status"] != "running":
            run = ROOT/"runs/phase1"/row["run_id"]
            checks = json.loads((run/"checks.json").read_text()) if (run/"checks.json").exists() else {}
            controls = checks.get("controls", [])
            row["physics_steps"] = sum(c["physics_steps"] for c in controls)
            row["simulated_seconds"] = row["physics_steps"]/120
            row["audits"] = [audit_episode(p) for p in sorted((ROOT/"datasets/phase1"/row["run_id"]/"episodes").glob("*.h5"))]
            row["audit_passed"] = bool(row["audits"]) and all(a["status"] == "consistent" for a in row["audits"])
            latencies = row["inference_seconds"]
            row["latency"] = {"count": len(latencies), "first": latencies[0], "mean": float(np.mean(latencies)),
                              "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))} if latencies else None
            row["replay_exceeded"] = []
            replay = checks.get("reconstruction")
            if replay:
                for name, values in replay["errors"].items():
                    for metric in ("max_abs", "rmse"):
                        bound = tolerances[metric].get(name)
                        if bound is None or values[metric] is None or values[metric] > bound:
                            row["replay_exceeded"].append({"name": name, "metric": metric, "value": values[metric], "bound": bound})
            if not row.get("task_success") or not row["audit_passed"]:
                failures.append({"seed": row["seed"], "run_id": row["run_id"], "outcome": row.get("outcome"),
                                 "episode_paths": [a["file"] for a in row["audits"]],
                                 "policy_inputs_and_chunks": str(run/"policy"), "video": str(run/"head_camera.mp4")})
        rows.append(row)
    report = {"status": manifest["status"], "summary": summarize(rows, len(manifest["scenes"]["seeds"])),
              "rows": rows, "failure_index": failures,
              "replay_scope": "development parents; four parents also used for v3 calibration; not an independent test success estimate"}
    source_correction = directory/"source_versions/history_mask_correction.json"
    if source_correction.exists():
        report["source_provenance_correction"] = json.loads(source_correction.read_text())
    (directory/"report.json").write_text(json.dumps(report, indent=2))
    fields = ["seed", "status", "reason", "task_success", "controls", "physics_steps", "simulated_seconds",
              "inferences", "inference_first_s", "inference_mean_s", "inference_p95_s", "replay_match", "audit_passed"]
    with (directory/"metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            latency = r.get("latency") or {}
            writer.writerow({"seed": r["seed"], "status": r["status"], "reason": r.get("outcome", {}).get("reason"),
                             "task_success": r.get("task_success"), "controls": r.get("controls"),
                             "physics_steps": r.get("physics_steps"), "simulated_seconds": r.get("simulated_seconds"),
                             "inferences": latency.get("count"), "inference_first_s": latency.get("first"),
                             "inference_mean_s": latency.get("mean"), "inference_p95_s": latency.get("p95"),
                             "replay_match": r.get("reconstruction", {}).get("valid_match"), "audit_passed": r.get("audit_passed")})
    print(json.dumps(report["summary"], indent=2))
    if any(r.get("audit_passed") is False for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
