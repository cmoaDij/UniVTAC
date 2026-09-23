"""Train-split-only calibration and provenance for recovery triggers.

The monitor consumes an observable risk proxy, while this module freezes the
mapping from that proxy to a trigger threshold.  Calibration records must be
from the train split; dev/test outcomes are never accepted as fitting input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "evotac.trigger_calibration.v1"
LABEL_SEMANTICS = "explicit_recovery_need.v1"


@dataclass(frozen=True)
class TriggerCalibration:
    split: str
    threshold: float
    contact_threshold: float
    min_budget_steps: int
    target_recall: float
    source_sha256: str
    sample_count: int
    positive_count: int
    label_semantics: str = LABEL_SEMANTICS
    version: str = SCHEMA

    def __post_init__(self):
        if self.version != SCHEMA:
            raise ValueError("unsupported trigger calibration schema")
        if self.label_semantics != LABEL_SEMANTICS:
            raise ValueError("trigger calibration requires explicit recovery labels")
        if self.split != "train":
            raise ValueError("trigger calibration must be fit on the train split")
        values = (self.threshold, self.contact_threshold, self.target_recall)
        if not np.isfinite(values).all() or any(not 0.0 <= float(v) <= 1.0 for v in values):
            raise ValueError("calibration thresholds and target recall must be in [0, 1]")
        if type(self.min_budget_steps) is not int or self.min_budget_steps < 1:
            raise ValueError("min_budget_steps must be a positive integer")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if type(self.positive_count) is not int or not 0 <= self.positive_count <= self.sample_count:
            raise ValueError("positive_count is invalid")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")

    def as_dict(self):
        return {"schema": self.version, "split": self.split,
                "threshold": self.threshold, "contact_threshold": self.contact_threshold,
                "min_budget_steps": self.min_budget_steps, "target_recall": self.target_recall,
                "source_sha256": self.source_sha256, "sample_count": self.sample_count,
                "positive_count": self.positive_count, "label_semantics": self.label_semantics}


def _threshold_for_recall(scores, labels, target_recall):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives < 1:
        raise ValueError("calibration needs at least one positive recovery outcome")
    candidates = np.unique(scores)
    candidates = np.concatenate(([0.0], candidates))
    rows = []
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        recall = tp / positives
        precision = tp / max(1, int(predicted.sum()))
        rows.append((float(threshold), recall, precision, fp, int(predicted.sum())))
    eligible = [row for row in rows if row[1] >= target_recall]
    if not eligible:
        eligible = rows
    # Prefer fewer false positives, then higher precision, then a lower
    # threshold so ties remain conservative and deterministic.
    return min(eligible, key=lambda row: (row[3], -row[2], row[0]))


def fit_trigger_calibration(records, *, source_sha256: str, target_recall=0.8,
                            min_budget_steps=12, contact_threshold=None):
    records = list(records)
    if not records:
        raise ValueError("at least one train calibration record is required")
    if not np.isfinite(target_recall) or not 0.0 < float(target_recall) <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    scores, labels = [], []
    for row in records:
        if row.get("split") != "train":
            raise ValueError("calibration records must all be from train")
        if "needs_recovery" not in row or type(row["needs_recovery"]) is not bool:
            raise ValueError("calibration records require an explicit boolean needs_recovery label")
        score = float(row["score"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("calibration scores must be finite and in [0, 1]")
        scores.append(score)
        labels.append(row["needs_recovery"])
    chosen = _threshold_for_recall(scores, labels, float(target_recall))
    return TriggerCalibration("train", chosen[0],
                              chosen[0] if contact_threshold is None else float(contact_threshold),
                              int(min_budget_steps), float(target_recall), source_sha256,
                              len(records), int(np.sum(labels)), LABEL_SEMANTICS)


def save_calibration(calibration: TriggerCalibration, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(calibration.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_calibration(path, *, expected_split=None):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported trigger calibration schema")
    calibration = TriggerCalibration(split=payload["split"], threshold=float(payload["threshold"]),
                                     contact_threshold=float(payload["contact_threshold"]),
                                     min_budget_steps=int(payload["min_budget_steps"]),
                                     target_recall=float(payload["target_recall"]),
                                     source_sha256=payload["source_sha256"],
                                     sample_count=int(payload["sample_count"]),
                                     positive_count=int(payload["positive_count"]),
                                     label_semantics=payload.get("label_semantics", ""),
                                     version=payload["schema"])
    if expected_split in {"dev", "test"} and calibration.split != "train":
        raise ValueError("dev/test evaluation must use a train-fitted trigger calibration")
    return calibration


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
