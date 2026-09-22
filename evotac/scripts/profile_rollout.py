"""Summarize wall-clock rates and GPU snapshots for one EvoTac dataset run."""
import argparse
import json
from pathlib import Path

import h5py

from evotac.data.rollout_logger import read_tree
from evotac.perf.runtime_probe import gpu_snapshot, summarize_rates


def read_executions(run_path):
    rows = []
    for path in sorted((Path(run_path) / "episodes").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            for key in sorted(handle["actions"], key=lambda value: int(value)):
                rows.append(read_tree(handle["actions"][key]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="dataset run directory containing episodes/")
    args = parser.parse_args()
    executions = read_executions(args.run)
    report = {"run": str(args.run), "rates": summarize_rates(executions),
              "gpu_snapshot_now": gpu_snapshot()}
    manifest = args.run / "manifest.json"
    if manifest.exists():
        report["manifest"] = json.loads(manifest.read_text())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
