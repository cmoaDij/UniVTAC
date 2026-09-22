"""Freeze development repeat statistics with an explicit float32 precision floor."""
import argparse
import json
from pathlib import Path

import numpy as np

from evotac.config import ROOT
from evotac.envs.state_replay import calibrate_tolerances, execution_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_run", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = json.loads((args.calibration_run/"checks.json").read_text())
    if checks["status"] != "calibrated_pending_independent_validation":
        raise ValueError("Calibration run did not complete")
    reports = [json.loads(line) for line in (args.calibration_run/"reconstruction.jsonl").read_text().splitlines()]
    parents = set(checks["development_parents"])
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT/"configs"):
        raise ValueError("Frozen tolerances must be under evotac/configs")
    if destination.exists():
        raise FileExistsError("Frozen tolerance versions are immutable; choose a new version")
    # This analytical floor prevents exact-zero sample variance from demanding
    # bit-identical float32 state. It is below one micrometre/microradian and never
    # applied to image errors. It does not use validation-parent measurements.
    tolerance = calibrate_tolerances(reports, parents, args.version, numeric_floor=8*np.finfo(np.float32).eps)
    run_id = args.calibration_run.name
    manifest = json.loads((ROOT/"datasets/phase1"/run_id/"manifest.json").read_text())
    tolerance.update(source_run=str(args.calibration_run.resolve()),
                     execution_fingerprint=execution_fingerprint(manifest["config"]),
                     precision_floor_rationale="8 * float32 epsilon in native state units; images use repeat statistics only",
                     status="frozen_pending_independent_validation")
    destination.write_text(json.dumps(tolerance, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
