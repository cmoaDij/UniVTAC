"""Summarize an opt-in UIPC reset trace without running Isaac Sim."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import ROOT
from evotac.data.rollout_logger import read_tree


def _error(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return {"compatible": False, "max_abs": None, "rmse": None}
    delta = a - b
    return {"compatible": True,
            "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0}


def analyze(path):
    path = Path(path)
    with h5py.File(path, "r") as handle:
        attempts = sorted(handle.keys(), key=int)
        if len(attempts) < 2:
            raise ValueError("trace must contain at least two attempts")
        left, right = handle[attempts[0]], handle[attempts[1]]
        stages = sorted(set(left) & set(right))
        report = {"trace": str(path), "attempts": attempts[:2], "stages": [],
                  "evidence_scope": "diagnostic only; no replay tolerance or training label is changed"}
        for stage in stages:
            a, b = read_tree(left[stage]), read_tree(right[stage])
            objects = {}
            for name in sorted(set(a.get("objects", {})) & set(b.get("objects", {}))):
                objects[name] = {field: _error(a["objects"][name][field], b["objects"][name][field])
                                 for field in ("position", "velocity")}
            robot = {name: _error(a["robot"][name], b["robot"][name])
                     for name in a.get("robot", {}) if name in b.get("robot", {})}
            max_object = max((value[metric]["max_abs"]
                              for value in objects.values() for metric in ("position", "velocity")
                              if value[metric]["max_abs"] is not None), default=0.0)
            max_robot = max((value["max_abs"] for value in robot.values()
                             if value["max_abs"] is not None), default=0.0)
            report["stages"].append({"stage": stage,
                                     "physics_step": a.get("physics_step"),
                                     "uipc_frame": a.get("uipc_frame"),
                                     "max_object_max_abs": max_object,
                                     "max_robot_max_abs": max_robot,
                                     "objects": objects,
                                     "robot": robot})
        report["first_object_divergence"] = next(
            (row for row in report["stages"] if row["max_object_max_abs"] > 0), None)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.trace)
    output = args.output or args.trace.with_name("reset_trace_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "first_object_divergence": report["first_object_divergence"]}, indent=2))


if __name__ == "__main__":
    main()
