"""Run the P5 paired acceptance gate on frozen baseline/candidate JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evotac.learning.continual_adaptation import (CandidateEvaluation,
                                                   ContinualAdaptationController)


def _rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty evidence file: {path}")
    indexed = {}
    for row in rows:
        key = (row.get("parent_scene_id"), int(row.get("seed", -1)))
        if not isinstance(key[0], str) or key[1] < 0 or key in indexed:
            raise ValueError("evidence needs unique parent_scene_id/seed keys")
        if row.get("split") != "acceptance":
            raise ValueError("P5 acceptance evidence must use split=acceptance")
        indexed[key] = row
    return indexed


def _evaluation(rows, version):
    required = ("new_condition_success", "old_condition_success")
    if any(not all(key in row for key in required) for row in rows.values()):
        raise ValueError("each acceptance row needs new and old condition success")
    new = tuple(float(row["new_condition_success"]) for row in rows.values())
    old = tuple(float(row["old_condition_success"]) for row in rows.values())
    new_violation = tuple(float(row["new_condition_violation"]) for row in rows.values()
                          if "new_condition_violation" in row)
    old_violation = tuple(float(row["old_condition_violation"]) for row in rows.values()
                          if "old_condition_violation" in row)
    if bool(new_violation) != bool(old_violation) or (new_violation and len(new_violation) != len(new)):
        raise ValueError("violation evidence must be present for every paired row")
    return CandidateEvaluation(version, new, old, new_violation, old_violation)


def accept(baseline_path, candidate_path, *, state_path=None, current_version=None,
           candidate_version, candidate_skill_count, condition):
    baseline = _rows(baseline_path)
    candidate = _rows(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate acceptance parents/seeds are not paired")
    if state_path is None:
        if current_version is None:
            raise ValueError("current_version is required without a controller state")
        controller = ContinualAdaptationController(current_version=current_version)
    else:
        controller = ContinualAdaptationController.load(state_path)
    result = controller.propose(
        condition=condition, candidate_version=candidate_version,
        candidate_skill_count=candidate_skill_count,
        current_samples=list(candidate.values()), retrieved_samples=list(baseline.values()),
        uniform_old_samples=list(baseline.values()), baseline=_evaluation(baseline, controller.current_version),
        candidate=_evaluation(candidate, candidate_version))
    if state_path is not None:
        controller.save(state_path)
    return {"status": "accepted" if result.accepted else "rejected", "condition": condition,
            "candidate_version": candidate_version, "reason": result.reason,
            "decision": vars(result.decision), "sampled_count": result.sampled_count,
            "controller_version": controller.current_version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--candidate-skill-count", required=True, type=int)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--current-version")
    args = parser.parse_args()
    result = accept(args.baseline, args.candidate, state_path=args.state,
                    current_version=args.current_version, candidate_version=args.candidate_version,
                    candidate_skill_count=args.candidate_skill_count, condition=args.condition)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
