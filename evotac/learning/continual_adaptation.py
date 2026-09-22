"""Candidate adaptation and acceptance gates with explicit paired evidence."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_version: str
    new_condition_success: tuple[float, ...]
    old_condition_success: tuple[float, ...]
    new_condition_violations: tuple[float, ...] = ()
    old_condition_violations: tuple[float, ...] = ()


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    new_lower_bound: float | None
    old_lower_bound: float | None
    violation_upper_bound: float | None


@dataclass(frozen=True)
class AdaptationRound:
    condition: str
    baseline_version: str
    candidate_version: str
    candidate_skill_count: int
    sampled_count: int
    accepted: bool
    reason: str
    decision: AcceptanceDecision


class ContinualAdaptationController:
    """Versioned P5 controller with explicit candidate acceptance.

    The controller owns bookkeeping and sample mixing only.  Candidate
    training and evaluation must supply real paired evidence; no result is
    inferred from a missing branch and no candidate is auto-accepted.
    """

    def __init__(self, *, current_version: str, skill_count=1, max_skill_count=4,
                 gate: "AcceptanceGate | None" = None,
                 ratios=(0.5, 0.25, 0.25), seed=0):
        if not isinstance(current_version, str) or not current_version:
            raise ValueError("current_version must be nonempty")
        if type(skill_count) is not int or skill_count < 1:
            raise ValueError("skill_count must be positive integer")
        if type(max_skill_count) is not int or max_skill_count < skill_count:
            raise ValueError("max_skill_count must be >= skill_count")
        if len(ratios) != 3 or any(float(value) < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError("ratios must be three nonnegative values summing to one")
        self.current_version = current_version
        self.skill_count = skill_count
        self.max_skill_count = max_skill_count
        self.gate = gate or AcceptanceGate()
        self.ratios = tuple(float(value) for value in ratios)
        self.seed = int(seed)
        self.history: list[AdaptationRound] = []

    def propose(self, *, condition: str, candidate_version: str, candidate_skill_count: int,
                current_samples, retrieved_samples, uniform_old_samples,
                baseline: CandidateEvaluation, candidate: CandidateEvaluation):
        if not isinstance(condition, str) or not condition:
            raise ValueError("condition must be nonempty")
        if not isinstance(candidate_version, str) or not candidate_version:
            raise ValueError("candidate_version must be nonempty")
        if candidate_version == self.current_version:
            raise ValueError("candidate version must differ from active version")
        if baseline.candidate_version != self.current_version:
            raise ValueError("baseline evidence is not for the active version")
        if candidate.candidate_version != candidate_version:
            raise ValueError("candidate evidence version does not match candidate_version")
        if type(candidate_skill_count) is not int or not 1 <= candidate_skill_count <= self.max_skill_count:
            raise ValueError("candidate skill count exceeds configured capacity")
        if candidate_skill_count > self.skill_count + 1:
            raise ValueError("one adaptation round may add at most one skill")
        if sum(row.condition == condition for row in self.history) >= 2:
            raise ValueError("at most two candidate evaluations are allowed per condition")
        sampled = adaptation_mix(current_samples, retrieved_samples, uniform_old_samples,
                                  ratios=self.ratios, seed=self.seed + len(self.history))
        decision = self.gate.evaluate(baseline, candidate)
        accepted = bool(decision.accepted)
        if accepted:
            self.current_version = candidate_version
            self.skill_count = candidate_skill_count
        round_result = AdaptationRound(condition, baseline.candidate_version, candidate_version,
                                       candidate_skill_count, len(sampled), accepted,
                                       decision.reason, decision)
        self.history.append(round_result)
        return round_result

    def state_dict(self):
        return {
            "schema": "evotac.continual_adaptation_controller.v1",
            "current_version": self.current_version,
            "skill_count": self.skill_count,
            "max_skill_count": self.max_skill_count,
            "ratios": self.ratios,
            "seed": self.seed,
            "gate": {"min_new_gain": self.gate.min_new_gain,
                     "max_old_drop": self.gate.max_old_drop,
                     "max_violation_increase": self.gate.max_violation_increase},
            "history": [{
                "condition": row.condition,
                "baseline_version": row.baseline_version,
                "candidate_version": row.candidate_version,
                "candidate_skill_count": row.candidate_skill_count,
                "sampled_count": row.sampled_count,
                "accepted": row.accepted,
                "reason": row.reason,
                "decision": vars(row.decision),
            } for row in self.history],
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), allow_nan=False, default=lambda value: value.value
                                   if hasattr(value, "value") else value), encoding="utf-8")

    @classmethod
    def load(cls, path, *, gate=None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "evotac.continual_adaptation_controller.v1":
            raise ValueError("unsupported continual adaptation controller schema")
        saved_gate = payload.get("gate", {})
        if gate is None and saved_gate:
            gate = AcceptanceGate(**{key: float(saved_gate[key]) for key in
                                     ("min_new_gain", "max_old_drop", "max_violation_increase")})
        controller = cls(current_version=payload["current_version"], skill_count=int(payload["skill_count"]),
                          max_skill_count=int(payload["max_skill_count"]), gate=gate,
                          ratios=tuple(payload["ratios"]), seed=int(payload["seed"]))
        for row in payload.get("history", []):
            decision = AcceptanceDecision(**row["decision"])
            controller.history.append(AdaptationRound(
                row["condition"], row["baseline_version"], row["candidate_version"],
                int(row["candidate_skill_count"]), int(row["sampled_count"]),
                bool(row["accepted"]), row["reason"], decision))
        return controller


def _mean_lower(values, z=1.96):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None
    return float(values.mean() - z * values.std(ddof=1) / np.sqrt(values.size))


def _mean_upper(values, z=1.96):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None
    return float(values.mean() + z * values.std(ddof=1) / np.sqrt(values.size))


class AcceptanceGate:
    """Default gate from the plan: new gain, old retention, violation cap."""

    def __init__(self, *, min_new_gain=0.0, max_old_drop=0.05, max_violation_increase=0.02):
        self.min_new_gain = float(min_new_gain)
        self.max_old_drop = float(max_old_drop)
        self.max_violation_increase = float(max_violation_increase)
        if not np.isfinite([self.min_new_gain, self.max_old_drop, self.max_violation_increase]).all():
            raise ValueError("acceptance thresholds must be finite")
        if self.max_old_drop < 0 or self.max_violation_increase < 0:
            raise ValueError("drop and violation thresholds must be nonnegative")

    def evaluate(self, baseline: CandidateEvaluation, candidate: CandidateEvaluation) -> AcceptanceDecision:
        if candidate.candidate_version == baseline.candidate_version:
            raise ValueError("baseline and candidate versions must differ")
        if len(candidate.new_condition_success) != len(baseline.new_condition_success) or len(candidate.old_condition_success) != len(baseline.old_condition_success):
            return AcceptanceDecision(False, "insufficient_paired_evidence", None, None, None)
        for values in (baseline.new_condition_success, candidate.new_condition_success,
                       baseline.old_condition_success, candidate.old_condition_success):
            array = np.asarray(values, dtype=np.float64)
            if np.any(~np.isfinite(array)) or np.any((array < 0) | (array > 1)):
                raise ValueError("success evidence must be finite probabilities in [0, 1]")
        new_delta = np.asarray(candidate.new_condition_success) - np.asarray(baseline.new_condition_success)
        old_delta = np.asarray(candidate.old_condition_success) - np.asarray(baseline.old_condition_success)
        new_lower = _mean_lower(new_delta)
        old_lower = _mean_lower(old_delta)
        if new_lower is None or old_lower is None:
            return AcceptanceDecision(False, "insufficient_paired_evidence", new_lower, old_lower, None)
        violation_upper = None
        if candidate.new_condition_violations or baseline.new_condition_violations:
            if (len(candidate.new_condition_violations) != len(baseline.new_condition_violations)
                    or len(candidate.new_condition_violations) != len(candidate.new_condition_success)):
                return AcceptanceDecision(False, "insufficient_paired_evidence", new_lower, old_lower, None)
            if any(np.any(~np.isfinite(np.asarray(values, dtype=np.float64))) or
                   np.any((np.asarray(values, dtype=np.float64) < 0) | (np.asarray(values, dtype=np.float64) > 1))
                   for values in (baseline.new_condition_violations, candidate.new_condition_violations)):
                raise ValueError("violation evidence must be finite probabilities in [0, 1]")
            violation_delta = np.asarray(candidate.new_condition_violations) - np.asarray(baseline.new_condition_violations)
            violation_upper = _mean_upper(violation_delta)
        if new_lower <= self.min_new_gain:
            return AcceptanceDecision(False, "new_condition_gain_not_proven", new_lower, old_lower, violation_upper)
        if old_lower < -self.max_old_drop:
            return AcceptanceDecision(False, "old_condition_retention_failed", new_lower, old_lower, violation_upper)
        if violation_upper is not None and violation_upper > self.max_violation_increase:
            return AcceptanceDecision(False, "violation_increase_failed", new_lower, old_lower, violation_upper)
        return AcceptanceDecision(True, "all_acceptance_gates_passed", new_lower, old_lower, violation_upper)


def adaptation_mix(current, retrieved, uniform_old, *, ratios=(0.5, 0.25, 0.25), seed=0):
    """Sample the planned 50/25/25 current/retrieved/uniform-old mixture."""
    groups = [list(current), list(retrieved), list(uniform_old)]
    if len(ratios) != 3 or any(float(value) < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three nonnegative values summing to one")
    if any(ratio > 0 and not group for ratio, group in zip(ratios, groups)):
        raise ValueError("all nonzero-ratio adaptation groups must contain samples")
    if not any(groups):
        return []
    # Choose a batch size from the available pool, then allocate it by the
    # requested ratios.  Sampling with replacement is explicit when a group
    # is smaller than its allocation; no group is silently dropped.
    total = max(3, max(len(group) for group in groups))
    rng = np.random.default_rng(seed)
    count = total
    raw = [count * float(ratio) for ratio in ratios]
    allocations = [int(np.floor(value)) for value in raw]
    remainder = count - sum(allocations)
    fractional_order = np.argsort([-(value - np.floor(value)) for value in raw])
    for index in fractional_order[:remainder]:
        allocations[int(index)] += 1
    selected = []
    for group, ratio, allocation in zip(groups, ratios, allocations):
        if ratio == 0 or allocation == 0:
            continue
        selected.extend(group[int(index)] for index in rng.integers(0, len(group), size=allocation))
    return selected
