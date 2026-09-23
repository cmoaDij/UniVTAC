"""Deployment-side trigger monitor using only observable feature summaries."""
from dataclasses import dataclass
import math

from evotac.learning.trigger_calibration import TriggerCalibration


@dataclass(frozen=True)
class MonitorDecision:
    trigger: bool
    reason: str
    score: float


class RecoveryMonitor:
    def __init__(self, *, object_lost_risk=0.8, contact_blocked=0.8, min_budget_steps=12,
                 calibration: TriggerCalibration | None = None):
        if calibration is not None and not isinstance(calibration, TriggerCalibration):
            raise TypeError("calibration must be a TriggerCalibration")
        if calibration is not None:
            object_lost_risk = calibration.threshold
            contact_blocked = calibration.contact_threshold
            min_budget_steps = calibration.min_budget_steps
        self.object_lost_risk = float(object_lost_risk)
        self.contact_blocked = float(contact_blocked)
        self.min_budget_steps = int(min_budget_steps)
        self.calibration = calibration

    @property
    def calibrated(self):
        return self.calibration is not None

    def provenance(self):
        return self.calibration.as_dict() if self.calibration is not None else {
            "schema": "evotac.trigger_proxy.v1", "calibrated": False,
            "threshold": self.object_lost_risk, "contact_threshold": self.contact_blocked,
            "min_budget_steps": self.min_budget_steps,
        }

    def decide(self, features: dict) -> MonitorDecision:
        remaining = int(features.get("remaining_physics_steps", 0))
        if remaining < self.min_budget_steps:
            return MonitorDecision(False, "insufficient_budget", 0.0)
        risks = {key: float(features.get(key, 0.0)) for key in ("object_lost_risk", "contact_blocked")}
        if any(not math.isfinite(value) for value in risks.values()):
            return MonitorDecision(False, "invalid_risk", 0.0)
        if risks["object_lost_risk"] >= self.object_lost_risk:
            return MonitorDecision(True, "object_lost_risk", risks["object_lost_risk"])
        if risks["contact_blocked"] >= self.contact_blocked:
            return MonitorDecision(True, "contact_blocked", risks["contact_blocked"])
        return MonitorDecision(False, "nominal", max(risks.values()))
