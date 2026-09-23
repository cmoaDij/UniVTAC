"""Evaluate one frozen recovery candidate on every declared dev parent."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evotac.config import ROOT, output_path
from evotac.learning.recovery_experiment import scene_splits, sha256, validate_parents


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/recovery_dev_comparison.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-from", type=Path,
                        help="create a new auditable run, carrying finished rows forward and retrying others")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("split") != "dev" or not protocol.get("seeds"):
        raise ValueError("candidate evaluation requires a nonempty dev protocol")
    validate_parents(protocol["seeds"], scene_splits(ROOT / "configs/phase1_insert_hole.yaml"), "evaluate", "dev")
    directory = output_path(Path("runs/phase1") / args.run_id)
    directory.mkdir(parents=True, exist_ok=args.resume)
    lock = (ROOT / "runs/phase1/recovery_dev_comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = [args.protocol, args.checkpoint, args.warmstart,
             ROOT / "configs/phase1_insert_hole.yaml", ROOT / "configs/ftp1_insert_hole_chunk16.yaml",
             ROOT / "configs/recovery_sac_minimal.yaml"]
    files += [p for group in ("scripts", "learning", "policy", "envs", "evaluation", "data", "perf")
              for p in (ROOT / group).rglob("*.py")]
    frozen = {str(p.resolve()): sha256(p) for p in files}
    status_path = directory / "status.json"
    if args.resume and args.retry_from:
        raise ValueError("--resume and --retry-from are mutually exclusive")
    if args.resume:
        state = json.loads(status_path.read_text())
        if state["frozen_files_sha256"] != frozen or state["gpu"] != args.gpu:
            raise ValueError("resume changed frozen inputs or GPU")
        if any(row["status"] not in {"pending", "finished"} for row in state["rows"]):
            raise ValueError("resume cannot overwrite an incomplete or failed attempt; use --retry-from")
    elif args.retry_from:
        source = json.loads(args.retry_from.joinpath("status.json").read_text())
        if source["frozen_files_sha256"] != frozen or source["gpu"] != args.gpu:
            raise ValueError("retry changed frozen inputs or GPU")
        source_rows = {int(row["seed"]): row for row in source["rows"]}
        state = {"status": "planned", "run_id": args.run_id, "protocol": protocol,
                 "gpu": args.gpu, "git_commit": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"], text=True).strip(),
                 "frozen_files_sha256": frozen, "retry_from": str(args.retry_from.resolve()),
                 "rows": [], "created_unix": time.time()}
        for seed in protocol["seeds"]:
            previous = source_rows.get(int(seed))
            if previous and previous.get("status") == "finished":
                state["rows"].append(previous)
            else:
                state["rows"].append({"seed": int(seed), "status": "pending",
                                      "previous_attempt": previous.get("run_id") if previous else None})
        atomic_json(directory / "protocol.json", state)
    else:
        state = {"status": "planned", "run_id": args.run_id, "protocol": protocol,
                 "gpu": args.gpu, "git_commit": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"], text=True).strip(),
                 "frozen_files_sha256": frozen,
                 "rows": [{"seed": int(seed), "status": "pending"} for seed in protocol["seeds"]],
                 "created_unix": time.time()}
        atomic_json(directory / "protocol.json", state)

    def save():
        state["updated_unix"] = time.time()
        state["finished"] = sum(row["status"] == "finished" for row in state["rows"])
        state["successes"] = sum(row.get("success", False) for row in state["rows"] if row["status"] == "finished")
        atomic_json(status_path, state)

    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMNI_KIT_ACCEPT_EULA="YES",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = str(Path(sys.executable).resolve().parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        for index, row in enumerate(state["rows"]):
            if row["status"] == "finished":
                continue
            if any(sha256(p) != digest for p, digest in frozen.items()):
                raise RuntimeError("frozen input changed during candidate evaluation")
            run_id = f"{args.run_id}_{index:02d}_candidate_s{row['seed']}"
            checks_path = ROOT / "runs/phase1" / run_id / "checks.json"
            if checks_path.parent.exists():
                raise RuntimeError(f"attempt directory already exists: {run_id}")
            command = [sys.executable, "-u", "-m", "evotac.scripts.train_recovery",
                       "--run-id", run_id, "--performance-profile", "--device", "cuda:0",
                       "--model-gpu", args.gpu, "--mode", "evaluate", "--split", "dev",
                       "--seeds", str(row["seed"]), "--history-checkpoint", str(args.warmstart.resolve()),
                       "--trainer-checkpoint", str(args.checkpoint.resolve()), "--threads", "4",
                       "--monitor-object-lost-risk", str(protocol["monitor_object_lost_risk"]),
                       "--max-recovery-actions", str(protocol["max_recovery_actions"]),
                       "--stable-cycles", str(protocol["stable_cycles"])]
            row.update(status="running", run_id=run_id, command=command, started_unix=time.time())
            state["status"] = "running"
            save()
            with (directory / f"{index:02d}.log").open("x") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                row["pid"] = child.pid
                save()
                row["exit_code"] = child.wait()
            checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
            if row["exit_code"] or checks.get("status") != "completed" or len(checks.get("episodes", [])) != 1:
                row.update(status="failed", error=checks.get("error", "incomplete child process"))
                raise RuntimeError(f"candidate attempt failed: {run_id}")
            episode = checks["episodes"][0]
            valid = episode["terminal_outcome"].get("valid_trial") is True and not episode["terminal_outcome"].get("incomplete", False)
            if (episode["seed"] != row["seed"] or episode["split"] != "dev" or episode["updates"]
                    or (valid and episode.get("evaluation_weights_unchanged") is not True)):
                raise RuntimeError("candidate evaluation provenance/weight invariance failed")
            row.update(status="finished", valid_trial=valid, success=valid and episode["reason"] == "success",
                       reason=episode["reason"], recovery_actions=episode["recovery_actions"],
                       episode_path=episode["episode_path"], checks_sha256=sha256(checks_path),
                       episode_sha256=sha256(episode["episode_path"]), finished_unix=time.time())
            save()
            print(json.dumps({"completed": index + 1, "total": len(state["rows"]),
                              "seed": row["seed"], "reason": row["reason"],
                              "recovery_actions": row["recovery_actions"]}), flush=True)
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
