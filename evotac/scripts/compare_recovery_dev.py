"""Frozen four-arm development comparison, one fresh Isaac process per attempt."""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from evotac.config import ROOT, output_path
from evotac.learning.recovery_experiment import sha256, scene_splits, validate_parents


ARMS = ("baseline_a", "baseline_b", "warmstart", "trained")


def schedule(protocol):
    if protocol["split"] != "dev" or tuple(protocol["arms"]) != ARMS:
        raise ValueError("Only the declared four dev arms are supported")
    seeds = list(protocol["seeds"])
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    rng = random.Random(protocol["order_seed"])
    rng.shuffle(seeds)
    order = list(ARMS)
    rng.shuffle(order)
    return [{"seed": seed, "arm": arm} for i, seed in enumerate(seeds)
            for arm in order[i % 4:] + order[:i % 4]]


def summarize(rows, expected_per_arm):
    result = {"scope": "exploratory dev only", "test_started": False,
              "strict_snapshot_audit_passed": False, "arms": {}, "comparisons": {}}
    for arm in ARMS:
        attempted = [r for r in rows if r["arm"] == arm and r.get("status") == "finished"]
        valid = [r for r in attempted if r.get("valid_trial") is True]
        successes = sum(r["success"] for r in valid)
        result["arms"][arm] = {"scheduled": expected_per_arm, "finished": len(attempted),
            "valid": len(valid), "invalid": len(attempted)-len(valid), "successes": successes,
            "success_rate_valid": successes/len(valid) if valid else None,
            "successes_per_scheduled_attempt": successes/expected_per_arm,
            "reasons": dict(Counter(r.get("reason") for r in attempted)),
            "triggered": sum(r.get("recovery_actions", 0) > 0 for r in valid),
            "recovery_actions": sum(r.get("recovery_actions", 0) for r in valid)}
    for candidate, reference in (("baseline_b", "baseline_a"), ("trained", "baseline_a"), ("trained", "warmstart")):
        lookup = {(r["arm"], r["seed"]): r for r in rows if r.get("valid_trial") is True}
        seeds = sorted(seed for arm, seed in lookup if arm == candidate and (reference, seed) in lookup)
        deltas = [int(lookup[candidate, s]["success"]) - int(lookup[reference, s]["success"]) for s in seeds]
        wins, losses = deltas.count(1), deltas.count(-1)
        discordant = wins + losses
        p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses)+1)) / 2**discordant)
        interval = None
        if deltas:
            samples = np.random.default_rng(20260923).choice(deltas, size=(20000, len(deltas))).mean(1)
            interval = np.quantile(samples, [.025, .975]).tolist()
        result["comparisons"][f"{candidate}_vs_{reference}"] = {
            "complete_dev_parents": len(seeds), "candidate_only_success": wins,
            "reference_only_success": losses, "outcome_disagreements": discordant,
            "success_rate_difference": float(np.mean(deltas)) if deltas else None,
            "parent_bootstrap_95_interval_descriptive": interval,
            "exact_mcnemar_two_sided_p_descriptive": p if deltas else None,
            "caveat": "same parent seeds are statistical blocks, not identical hidden-state counterfactuals; small dev sample; no confirmatory inference"}
    return result


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/recovery_dev_comparison.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT.parent)
    protocol = json.loads(args.protocol.read_text())
    planned = schedule(protocol)
    validate_parents(protocol["seeds"], scene_splits(ROOT / "configs/phase1_insert_hole.yaml"), "evaluate", "dev")
    directory = output_path(Path("runs/phase1") / args.run_id)
    directory.mkdir(parents=True, exist_ok=args.resume)
    # Shared across instances of this coordinator. Per-child GPU preflight also
    # refuses unrelated users of the device before starting Isaac.
    lock = (ROOT / "runs/phase1/recovery_dev_comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = [args.protocol, args.checkpoint, args.warmstart,
             ROOT / "configs/phase1_insert_hole.yaml", ROOT / "configs/ftp1_insert_hole_chunk16.yaml",
             ROOT / "configs/recovery_sac_minimal.yaml"]
    files += [p for group in ("scripts", "learning", "policy", "envs", "evaluation", "data", "perf")
              for p in (ROOT/group).rglob("*.py")]
    frozen = {str(p.resolve()): sha256(p) for p in files}
    path = directory / "status.json"
    if args.resume:
        state = json.loads(path.read_text())
        if state["frozen_files_sha256"] != frozen or state["gpu"] != args.gpu:
            raise ValueError("Resume must preserve protocol, weights, code and GPU")
        if any(r["status"] not in {"pending", "finished"} for r in state["rows"]):
            raise ValueError("Inspect incomplete attempt before resuming; never replace it silently")
    else:
        state = {"status": "planned", "protocol": protocol, "gpu": args.gpu,
                 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "frozen_files_sha256": frozen, "rows": [{**r, "status": "pending"} for r in planned],
                 "created_unix": time.time()}
        atomic_json(directory / "protocol.json", state)
    def save():
        state.update(updated_unix=time.time(), pid=os.getpid(), summary=summarize(state["rows"], len(protocol["seeds"])))
        atomic_json(path, state)
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMNI_KIT_ACCEPT_EULA="YES",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = str(Path(sys.executable).resolve().parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        for index, row in enumerate(state["rows"]):
            if row["status"] == "finished":
                continue
            if any(sha256(p) != digest for p, digest in frozen.items()):
                raise RuntimeError("Frozen source/config/weights changed during comparison")
            run_id = f"{args.run_id}_{index:02d}_{row['arm']}_s{row['seed']}"
            checks_path = ROOT / "runs/phase1" / run_id / "checks.json"
            if checks_path.parent.exists():
                raise RuntimeError(f"Attempt directory already exists: {run_id}")
            command = [sys.executable, "-u", "-m", "evotac.scripts.train_recovery", "--run-id", run_id,
                "--performance-profile", "--device", "cuda:0", "--model-gpu", args.gpu,
                "--mode", "evaluate", "--split", "dev", "--seeds", str(row["seed"]),
                "--history-checkpoint", str(args.warmstart.resolve()), "--threads", "4",
                "--monitor-object-lost-risk", str(protocol["monitor_object_lost_risk"]),
                "--max-recovery-actions", str(protocol["max_recovery_actions"]),
                "--stable-cycles", str(protocol["stable_cycles"])]
            if row["arm"] != "warmstart":
                command += ["--trainer-checkpoint", str(args.checkpoint.resolve())]
            if row["arm"].startswith("baseline"):
                command += ["--baseline"]
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
                raise RuntimeError(f"Attempt failed: {run_id}")
            episode = checks["episodes"][0]
            valid = episode["terminal_outcome"].get("valid_trial") is True and episode["terminal_outcome"].get("incomplete") is False
            if (episode["seed"] != row["seed"] or episode["split"] != "dev" or episode["updates"]
                    or (valid and episode.get("evaluation_weights_unchanged") is not True)):
                raise RuntimeError("Evaluation provenance/weight invariance failed")
            row.update(status="finished", valid_trial=valid, success=valid and episode["reason"] == "success",
                reason=episode["reason"], recovery_actions=episode["recovery_actions"],
                episode_path=episode["episode_path"], checks_sha256=sha256(checks_path),
                episode_sha256=sha256(episode["episode_path"]), finished_unix=time.time())
            save()
            print(json.dumps({"completed": index+1, "total": len(state["rows"]), "seed": row["seed"],
                              "arm": row["arm"], "reason": row["reason"], "recovery_actions": row["recovery_actions"]}), flush=True)
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
