"""Deployment-side trigger monitor using only observable feature summaries."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorDecision:
    trigger: bool
    reason: str
    score: float


class RecoveryMonitor:
    def __init__(self, *, object_lost_risk=0.8, contact_blocked=0.8, min_budget_steps=12):
        self.object_lost_risk = float(object_lost_risk)
        self.contact_blocked = float(contact_blocked)
        self.min_budget_steps = int(min_budget_steps)

    def decide(self, features: dict) -> MonitorDecision:
        remaining = int(features.get("remaining_physics_steps", 0))
        if remaining < self.min_budget_steps:
            return MonitorDecision(False, "insufficient_budget", 0.0)
        risks = {key: float(features.get(key, 0.0)) for key in ("object_lost_risk", "contact_blocked")}
        if risks["object_lost_risk"] >= self.object_lost_risk:
            return MonitorDecision(True, "object_lost_risk", risks["object_lost_risk"])
        if risks["contact_blocked"] >= self.contact_blocked:
            return MonitorDecision(True, "contact_blocked", risks["contact_blocked"])
        return MonitorDecision(False, "nominal", max(risks.values()))
