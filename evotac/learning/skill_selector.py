"""Effect-aware skill selection and versioned nearest-neighbour memory."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Selection:
    skill_name: str | None
    score: float
    reason: str
    candidates: tuple[dict, ...]


@dataclass(frozen=True)
class EffectRecord:
    embedding: np.ndarray
    skill_name: str
    success: float
    cost: float
    violation: float
    version: str
    valid: bool = True

    def __post_init__(self):
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError("effect memory embedding must be finite vector")
        if not np.isfinite([self.success, self.cost, self.violation]).all():
            raise ValueError("effect memory targets must be finite")
        if not 0.0 <= float(self.success) <= 1.0 or not 0.0 <= float(self.violation) <= 1.0:
            raise ValueError("success and violation must be probabilities in [0, 1]")
        if float(self.cost) < 0.0:
            raise ValueError("effect cost must be nonnegative")
        if not self.skill_name or not self.version:
            raise ValueError("skill_name and version are required")
        object.__setattr__(self, "embedding", embedding.copy())
        object.__setattr__(self, "success", float(self.success))
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "violation", float(self.violation))


class EffectMemory:
    def __init__(self, capacity=10000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be positive integer")
        self.capacity = capacity
        self.records: list[EffectRecord] = []

    def add(self, record: EffectRecord):
        if self.records and record.embedding.shape != self.records[0].embedding.shape:
            raise ValueError("effect embeddings must share a shape")
        if len(self.records) >= self.capacity:
            self.records.pop(0)
        self.records.append(record)

    def retrieve(self, embedding, k=20, *, version=None):
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1 or not np.isfinite(query).all():
            raise ValueError("effect query embedding must be a finite vector")
        if self.records and query.shape != self.records[0].embedding.shape:
            raise ValueError("effect query embedding shape does not match memory")
        compatible = [row for row in self.records if row.valid and (version is None or row.version == version)]
        compatible.sort(key=lambda row: float(np.linalg.norm(row.embedding - query)))
        return compatible[:max(0, int(k))]

    def state_dict(self):
        return {"schema": "evotac.effect_memory.v1", "capacity": self.capacity,
                "records": [{"embedding": row.embedding.tolist(), "skill_name": row.skill_name,
                             "success": row.success, "cost": row.cost, "violation": row.violation,
                             "version": row.version, "valid": row.valid} for row in self.records]}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "evotac.effect_memory.v1":
            raise ValueError("unsupported effect memory schema")
        memory = cls(int(payload["capacity"]))
        for row in payload.get("records", []):
            memory.add(EffectRecord(**row))
        return memory


class SkillSelector:
    def __init__(self, skill_names, *, cost_weight=0.1, violation_weight=0.5, min_success=0.0):
        self.skill_names = tuple(skill_names)
        if not self.skill_names:
            raise ValueError("at least one skill is required")
        self.cost_weight, self.violation_weight, self.min_success = map(float, (cost_weight, violation_weight, min_success))
        if (not np.isfinite([self.cost_weight, self.violation_weight, self.min_success]).all()
                or self.cost_weight < 0 or self.violation_weight < 0
                or not 0 <= self.min_success <= 1):
            raise ValueError("invalid effect selection weights or success threshold")

    def select(self, predictions: dict[str, dict], *, available=None) -> Selection:
        available = set(self.skill_names if available is None else available)
        candidates = []
        for name in self.skill_names:
            if name not in available or name not in predictions:
                continue
            values = predictions[name]
            success = float(values["success_probability"])
            score = success - self.cost_weight * float(values.get("cost", 0.0)) - self.violation_weight * float(values.get("violation_probability", 0.0))
            candidates.append({"skill_name": name, "score": score, "success_probability": success})
        candidates.sort(key=lambda row: row["score"], reverse=True)
        if not candidates:
            return Selection(None, float("-inf"), "no_compatible_skill", ())
        best = candidates[0]
        if best["success_probability"] < self.min_success:
            return Selection(None, best["score"], "all_below_success_threshold", tuple(candidates))
        return Selection(best["skill_name"], best["score"], "highest_effect_score", tuple(candidates))
