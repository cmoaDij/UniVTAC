"""Explicit runtime state machine for pausing the base policy and recovering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from evotac.learning.runtime_monitor import RecoveryMonitor


class RecoveryState(str, Enum):
    NORMAL = "normal"
    SELECT = "select"
    RECOVERY = "recovery"
    HANDOFF = "handoff"
    STOP = "stop"


@dataclass(frozen=True)
class StateEvent:
    previous: RecoveryState
    current: RecoveryState
    reason: str


class RecoveryStateMachine:
    """Small deterministic state machine; selection and control stay separate."""

    def __init__(self, monitor: RecoveryMonitor | None = None, *, stable_cycles=3,
                 min_recovery_cycles=1):
        if type(stable_cycles) is not int or stable_cycles < 1:
            raise ValueError("stable_cycles must be a positive integer")
        if type(min_recovery_cycles) is not int or min_recovery_cycles < 1:
            raise ValueError("min_recovery_cycles must be a positive integer")
        self.monitor = monitor or RecoveryMonitor()
        self.stable_cycles = stable_cycles
        self.min_recovery_cycles = min_recovery_cycles
        self.state = RecoveryState.NORMAL
        self.recovery_cycles = 0
        self.stable_count = 0

    def reset(self):
        self.state = RecoveryState.NORMAL
        self.recovery_cycles = 0
        self.stable_count = 0

    def _move(self, state, reason):
        event = StateEvent(self.state, state, reason)
        self.state = state
        return event

    def observe(self, features: dict, *, selected_skill: str | None = None,
                skill_available=True, action_executed=True) -> StateEvent | None:
        """Consume one observable cycle and return a transition when one occurs."""
        if self.state is RecoveryState.STOP:
            return None
        if not action_executed:
            previous = self.state
            self.state = RecoveryState.STOP
            return StateEvent(previous, RecoveryState.STOP, "action_not_executed")
        if self.state is RecoveryState.NORMAL:
            decision = self.monitor.decide(features)
            if decision.trigger:
                return self._move(RecoveryState.SELECT, decision.reason)
        elif self.state is RecoveryState.SELECT:
            if selected_skill and skill_available:
                self.recovery_cycles = 0
                self.stable_count = 0
                return self._move(RecoveryState.RECOVERY, f"skill_selected:{selected_skill}")
            if not skill_available or int(features.get("remaining_physics_steps", 0)) < self.monitor.min_budget_steps:
                return self._move(RecoveryState.STOP, "no_reliable_skill")
        elif self.state is RecoveryState.RECOVERY:
            self.recovery_cycles += 1
            if bool(features.get("hard_stop", False)):
                return self._move(RecoveryState.STOP, "hard_stop")
            if bool(features.get("recovery_budget_exhausted", False)):
                return self._move(RecoveryState.STOP, "recovery_budget_exhausted")
            if bool(features.get("stable", False)):
                self.stable_count += 1
            else:
                self.stable_count = 0
            if self.recovery_cycles >= self.min_recovery_cycles and self.stable_count >= self.stable_cycles:
                return self._move(RecoveryState.HANDOFF, "stable_for_handoff")
        elif self.state is RecoveryState.HANDOFF:
            return self._move(RecoveryState.NORMAL, "handoff_complete")
        return None
