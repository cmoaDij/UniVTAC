"""Freeze a train-only trigger calibration from an auditable JSON record file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evotac.learning.trigger_calibration import file_sha256, fit_trigger_calibration, save_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--min-budget-steps", type=int, default=12)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("rows", records.get("records"))
    if not isinstance(records, list):
        raise ValueError("input must be a JSON list or an object containing rows/records")
    calibration = fit_trigger_calibration(records, source_sha256=file_sha256(args.input),
                                          target_recall=args.target_recall,
                                          min_budget_steps=args.min_budget_steps)
    save_calibration(calibration, args.output)
    print(json.dumps(calibration.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
