"""Audit a replay report against a frozen tolerance file.

This deliberately separates numeric tolerance acceptance from the replay
version gate.  An implementation-drift replay can be numerically inside a
new candidate tolerance while still being ineligible for a strict historical
match; both facts are retained in the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _audit_report(report, tolerances):
    failures = []
    missing = []
    for key, error in report.get("errors", {}).items():
        max_limit = tolerances.get("max_abs", {}).get(key)
        rmse_limit = tolerances.get("rmse", {}).get(key)
        if max_limit is None or rmse_limit is None:
            missing.append(key)
            continue
        if error.get("max_abs") is None or error.get("rmse") is None:
            failures.append({"key": key, "reason": "incompatible_error"})
            continue
        if error["max_abs"] > max_limit or error["rmse"] > rmse_limit:
            failures.append({"key": key, "max_abs": error["max_abs"],
                              "max_abs_limit": max_limit, "rmse": error["rmse"],
                              "rmse_limit": rmse_limit})
    return {"pass": not failures and not missing, "failures": failures, "missing": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", type=Path, help="replay checks.json")
    parser.add_argument("tolerances", type=Path, help="frozen tolerance JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = json.loads(args.checks.read_text())
    tolerances = json.loads(args.tolerances.read_text())
    reconstruction = checks.get("reconstruction") or {}
    repeats = checks.get("current_repeats") or []
    result = {
        "status": "candidate_tolerance_audit",
        "checks": str(args.checks),
        "tolerances": str(args.tolerances),
        "tolerance_version": tolerances.get("version"),
        "hardware_comparison": checks.get("hardware_comparison"),
        "version_comparison": checks.get("version_comparison"),
        "replay_status": reconstruction.get("status"),
        "versions_match": reconstruction.get("versions_match"),
        "observable_match": reconstruction.get("observable_match"),
        "strict_valid_match": reconstruction.get("valid_match"),
        "source_error_thresholds": _audit_report(reconstruction, tolerances),
        "same_code_repeats": [],
    }
    for item in repeats:
        report = item.get("reconstruction") or {}
        result["same_code_repeats"].append({
            "index": item.get("index"),
            "error_thresholds": _audit_report(report, tolerances),
            "strict_valid_match": report.get("valid_match"),
            "observable_match": report.get("observable_match"),
            "same_code_tree_exact": (item.get("same_code_as_reference") or {}).get("exact_match"),
        })
    result["candidate_numeric_validation"] = bool(
        result["source_error_thresholds"]["pass"]
        and all(item["error_thresholds"]["pass"] for item in result["same_code_repeats"])
    )
    output = args.output or args.checks.with_name("independent_tolerance_validation.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(output)
    print(json.dumps({k: result[k] for k in ("candidate_numeric_validation", "strict_valid_match", "observable_match")}, indent=2))


if __name__ == "__main__":
    main()
