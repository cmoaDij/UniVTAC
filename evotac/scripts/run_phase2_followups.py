"""Run serial GPU acceptance/ablations after the live baseline supervisor exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evotac.config import ROOT, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sim-gpu", default="0")
    parser.add_argument("--model-gpu", default="9")
    args = parser.parse_args()
    directory = output_path(Path("runs/phase2")/args.run_id)
    directory.mkdir(parents=True, exist_ok=False)
    record = {"status": "starting", "command": sys.argv, "pid": os.getpid(), "steps": []}
    def save():
        (directory/"status.json").write_text(json.dumps(record, indent=2))
    save()
    if args.wait_pid:
        # Kernel process start time guards against PID reuse. This Conda build
        # lacks os.pidfd_open; a manifest alone is never treated as a live job.
        stat = Path(f"/proc/{args.wait_pid}/stat")
        try:
            started = stat.read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            started = None
        if started is not None:
            command = Path(f"/proc/{args.wait_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "evotac.scripts.run_phase2_baseline" not in command or args.baseline_id not in command:
                raise ValueError("Wait PID does not identify the requested baseline supervisor")
            record.update(status="waiting_for_baseline", wait_pid=args.wait_pid)
            save()
            while True:
                try:
                    current = stat.read_text().rsplit(")", 1)[1].split()
                except FileNotFoundError:
                    break
                if current[19] != started or current[0] == "Z":
                    break
                time.sleep(5)
    baseline = json.loads((ROOT/"runs/phase2"/args.baseline_id/"manifest.json").read_text())
    if baseline["status"] != "complete":
        record["status"] = "baseline_not_complete"
        save()
        raise SystemExit(1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.sim_gpu, OMNI_KIT_ACCEPT_EULA="YES",
               LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parent.parent/"lib")+":"+os.environ.get("LD_LIBRARY_PATH", ""))
    def run(label, module, arguments, checks_path=None, expected=None):
        command = [sys.executable, "-u", "-m", module, *arguments]
        row = {"label": label, "command": command, "status": "running"}
        record["steps"].append(row)
        record["status"] = "running"
        with (directory/f"{label}.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            save()
            row["exit_code"] = process.wait()
        passed = row["exit_code"] == 0
        if checks_path:
            check = json.loads(checks_path.read_text()) if checks_path.exists() else {}
            row["result_status"] = check.get("status")
            passed &= check.get("status") in expected
        row["status"] = "passed" if passed else "failed"
        save()
        if not passed:
            record["status"] = "stopped_on_failure"
            save()
            raise SystemExit(1)
    run("baseline_report", "evotac.scripts.report_phase2_baseline", ["--run-id", args.baseline_id])
    repeat_id = args.run_id+"_seed24_provenance_repeat"
    run("seed24_provenance_repeat", "evotac.scripts.run_ftp1",
        ["--livestream", "2", "--device", "cuda:0", "--model-gpu", args.model_gpu, "--seed", "24", "--run-id", repeat_id,
         "--policy-config", str(ROOT/"runs/phase2"/args.baseline_id/"policy.yaml"), "--max-controls", "200", "--replay-at", "12"],
        ROOT/"runs/phase1"/repeat_id/"checks.json", {"executed"})
    run("seed24_repeat_report", "evotac.scripts.report_ftp1_run", ["--run-id", repeat_id])
    run("seed24_repeat_audit", "evotac.scripts.audit_rollouts",
        [str(ROOT/"datasets/phase1"/repeat_id), "--output", str(directory/"seed24_repeat_audit.json")])
    handoff_id = args.run_id+"_handoff"
    run("handoff", "evotac.scripts.verify_ftp1_handoff",
        ["--livestream", "2", "--device", "cuda:0", "--model-gpu", args.model_gpu, "--seed", "0", "--run-id", handoff_id],
        ROOT/"runs/phase1"/handoff_id/"checks.json", {"passed"})
    run("handoff_audit", "evotac.scripts.audit_rollouts",
        [str(ROOT/"datasets/phase1"/handoff_id), "--output", str(directory/"handoff_audit.json")])
    grip_id = args.run_id+"_hold_gripper_seed0"
    run("hold_gripper", "evotac.scripts.run_ftp1",
        ["--livestream", "2", "--device", "cuda:0", "--model-gpu", args.model_gpu, "--seed", "0", "--run-id", grip_id,
         "--policy-config", str(ROOT/"configs/ftp1_insert_hole_hold_gripper.yaml"), "--max-controls", "200"],
        ROOT/"runs/phase1"/grip_id/"checks.json", {"executed"})
    run("gripper_report", "evotac.scripts.report_ftp1_run", ["--run-id", grip_id])
    run("gripper_diagnosis", "evotac.scripts.diagnose_ftp1_run", ["--run-id", grip_id])
    chunk_id = args.run_id+"_chunk16"
    run("chunk16_baseline", "evotac.scripts.run_phase2_baseline",
        ["--run-id", chunk_id, "--sim-gpu", args.sim_gpu, "--model-gpu", args.model_gpu,
         "--policy-config", str(ROOT/"configs/ftp1_insert_hole_chunk16.yaml"), "--replay-at", "0"],
        ROOT/"runs/phase2"/chunk_id/"manifest.json", {"complete"})
    run("chunk16_report", "evotac.scripts.report_phase2_baseline", ["--run-id", chunk_id])
    record["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
