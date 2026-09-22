"""Run isolated livestream acceptance jobs serially and retain authoritative results.

This parent process does not import Isaac. It retains return codes even when
Kit overrides a child process's Python exit/exception path. Check each stage's
checks.json or outcome report as well as its process return code.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evotac.config import ROOT, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    output = output_path(Path("runs/phase1") / args.suite_id)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    common = ["--livestream", "2", "--device", "cuda:0"]
    stages = [
        ("axes", ["-m", "evotac.scripts.check_action_axes", "--run-id", args.suite_id+"_axes", *common]),
        ("failures", ["-m", "evotac.scripts.smoke_failures", "--run-id", args.suite_id+"_failures", *common]),
        ("legacy", ["-m", "evotac.scripts.check_legacy_compatibility", "--episodes", "0", "1", "--run-id", args.suite_id+"_legacy", *common]),
        ("original_collection", ["scripts/collect_data.py", "insert_hole", "demo", "--start_seed", "0", "--max_seed", "0", "--gpu", args.gpu, *common,
                                 "--config-overrides", "collect_settings.episode_num=1", f"collect_settings.save_root_dir={output/'original_collection'}"]),
        ("calibration", ["-m", "evotac.scripts.calibrate_replay", "--fresh-processes", "--config", str(ROOT/"configs/calibration_insert_hole.yaml"), "--dev-seeds", "0", "18", "23", "24", "--repeats", "2", "--run-id", args.suite_id+"_calibration", *common]),
    ]
    results = []
    for name, arguments in stages:
        command = [sys.executable, "-u", *arguments]
        print(f"Starting {name}", flush=True)
        with (output/f"{name}.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
        row = {"stage": name, "command": command, "returncode": completed.returncode, "log": str(output/f"{name}.log")}
        stage_output = ROOT/"runs/phase1"/(args.suite_id+"_"+name)
        for report in ("checks.json", "legacy_compatibility.json"):
            if (stage_output/report).exists():
                row["report"] = str(stage_output/report)
                row["result"] = json.loads((stage_output/report).read_text())
        results.append(row)
        (output/"suite_results.json").write_text(json.dumps(results, indent=2))
        print(f"Finished {name}: process={completed.returncode}, report={row.get('result', {}).get('status', 'inspect outcome report')}", flush=True)
    print(f"Suite records: {output/'suite_results.json'}", flush=True)


if __name__ == "__main__":
    main()
