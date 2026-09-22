"""Resume real collection, calibrate on dev, then enforce the causal audit gate.

Run from the repository root in the UniVTAC environment. All subprocesses use
one physical GPU sequentially; test parents never enter calibration/training.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evotac.config import ROOT
from evotac.learning.recovery_experiment import resolved_split, scene_splits, sha256


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-run", default="p4_real_train_resume60_20260922")
    parser.add_argument("--run-id", default="p4_validation_resume_20260922")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--max-extra-batches", type=int, default=4)
    args = parser.parse_args()
    if not 500 <= args.target <= 2000 or not 0 <= args.max_extra_batches <= 10:
        raise ValueError("target must be 500..2000; max-extra-batches 0..10")
    os.chdir(ROOT.parent)
    runs = ROOT / "runs/phase1"
    folder = runs / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "process.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "waiting_for_collection", "source_run": args.source_run,
             "target_transitions": args.target, "pid": os.getpid(),
             "completed_runs": [], "calibration": [],
             "test_status": "not_started", "causal_evidence": False}

    def persist():
        state["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(folder / "status.json", state)

    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES",
               CUDA_VISIBLE_DEVICES="GPU-d86528b1-67d4-d67f-db2c-ccae580a1267",
               LD_LIBRARY_PATH="/data/ZED/conda/envs/UniVTAC/lib",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    warm = runs / "recovery_warmstart_train_seed1_6_9_10_20260921.pt"
    common = ["--performance-profile", "--device", "cuda:0", "--model-gpu", "8"]
    training_common = common + ["--batch-size", "32", "--learning-starts", "32",
                                "--threads", "4", "--history-checkpoint", str(warm)]

    def run(module, name, options):
        checks_path = runs / name / "checks.json"
        old = read(checks_path)
        if old and old.get("status") == "completed":
            return old
        if checks_path.parent.exists():
            raise RuntimeError(f"incomplete run exists; inspect before retry: {name}")
        state["active_run"] = name
        command = [sys.executable, "-u", "-m", module, "--run-id", name] + options
        state["active_command"] = command
        persist()
        with (folder / (name + ".log")).open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["active_pid"] = child.pid
            persist()
            code = child.wait()
        result = read(checks_path)
        if code or result is None or result.get("status") == "failed":
            raise RuntimeError(f"{name} failed (exit={code}); inspect log and checks")
        state["completed_runs"].append(name)
        persist()
        return result

    try:
        deadline = time.monotonic() + 12 * 3600
        while True:
            source = read(runs / args.source_run / "checks.json")
            if source:
                state["episodes_completed"] = len(source.get("episodes", []))
                state["transitions"] = source.get("training_evidence", {}).get("new_transitions", 0)
                persist()
                if source.get("status") == "completed":
                    break
                if source.get("status") == "failed":
                    raise RuntimeError(f"source collection failed: {source.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("source collection did not finish within 12 hours")
            time.sleep(30)

        checkpoint = runs / args.source_run / "recovery_trainer.pt"
        splits = scene_splits(ROOT / "configs/phase1_insert_hole.yaml")
        evidence = source["training_evidence"]
        used = {e["seed"] for e in evidence["episodes"]}
        for batch in range(args.max_extra_batches):
            if evidence["new_transitions"] >= args.target:
                break
            seeds = [seed for seed in range(139, 2000)
                     if seed not in used and resolved_split(seed, splits) == "train"][:60]
            if len(seeds) != 60:
                raise RuntimeError("not enough unused train parents")
            state["status"] = "collecting_additional_train_parents"
            name = f"{args.run_id}_extra{batch}"
            result = run("evotac.scripts.train_recovery", name, training_common + [
                "--mode", "train", "--split", "train", "--monitor-object-lost-risk", "0.3",
                "--trainer-checkpoint", str(checkpoint), "--seeds", *map(str, seeds)])
            checkpoint = runs / name / "recovery_trainer.pt"
            evidence = result["training_evidence"]
            used.update(seeds)
            state["transitions"] = evidence["new_transitions"]
            persist()
        if evidence["new_transitions"] < args.target:
            state["status"] = "insufficient_data_after_bounded_collection"
            persist()
            return

        state.update(status="calibrating_dev_only", frozen_checkpoint=str(checkpoint),
                     checkpoint_sha256=sha256(checkpoint))
        eval_common = training_common + ["--mode", "evaluate", "--split", "dev",
                                        "--trainer-checkpoint", str(checkpoint),
                                        "--seeds", "0", "18", "23"]
        baseline = run("evotac.scripts.train_recovery", args.run_id + "_dev_baseline",
                       eval_common + ["--baseline", "--monitor-object-lost-risk", "0.3"])
        state["dev_baseline_successes"] = sum(e["reason"] == "success" for e in baseline["episodes"])
        # Small prespecified grid, including the original 20-action protocol.
        for index, (threshold, cap) in enumerate(((.3, None), (.25, 12), (.3, 6), (.3, 12), (.35, 12))):
            options = eval_common + ["--monitor-object-lost-risk", str(threshold)]
            if cap is not None:
                options += ["--max-recovery-actions", str(cap)]
            result = run("evotac.scripts.train_recovery", f"{args.run_id}_dev_candidate{index}", options)
            eps = result["episodes"]
            row = {"threshold": threshold, "cap": cap, "n": len(eps),
                   "successes": sum(e["reason"] == "success" for e in eps),
                   "actions": sum(e["recovery_actions"] for e in eps),
                   "eligible": len(eps) == 3 and all(e["terminal_outcome"].get("valid_trial") is True
                                and not e["terminal_outcome"].get("incomplete", True) for e in eps)}
            state["calibration"].append(row)
            persist()
        eligible = [row for row in state["calibration"] if row["eligible"]]
        if not eligible:
            raise RuntimeError("no candidate completed all three dev trials validly")
        selected = sorted(eligible, key=lambda row: (-row["successes"], row["actions"]))[0]
        state.update(selected=selected, status="snapshot_diagnostic")
        # Calibration remains on dev; never debug/select on test seed12.
        audit = run("evotac.scripts.run_recovery_paired_eval", args.run_id + "_snapshot_dev23",
                    common + ["--seed", "23", "--split", "dev", "--prefix-controls", "12",
                              "--post-prefix-controls", "188", "--recovery-controls", str(selected["cap"] or 19),
                              "--warmstart", str(warm), "--trainer-checkpoint", str(checkpoint)])
        state["snapshot_paired_valid"] = audit.get("paired_valid") is True
        # A failed diagnostic never authorizes a causal claim or silent test run.
        state["status"] = "awaiting_causal_protocol_review" if state["snapshot_paired_valid"] else "blocked_snapshot_audit"
        state["test_status"] = "not_started_pending_causal_gate"
        persist()
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        persist()
        raise


if __name__ == "__main__":
    main()
