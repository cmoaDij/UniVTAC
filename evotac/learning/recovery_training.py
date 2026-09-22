"""Online recovery collection driver built around the real wrapper boundary.

The driver contains no Isaac imports and does not invent counterfactual labels.
The supplied base policy and recovery policy receive deployment observations;
the wrapper remains the only owner of physics and terminal outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from evotac.learning.recovery_rollout import CollectionResult, RecoveryCollector
from evotac.learning.runtime_monitor import RecoveryMonitor


@dataclass(frozen=True)
class TrainingEpisodeResult:
    status: str
    reason: str
    recovery_actions: int
    admitted_transitions: int
    terminal_outcome: dict


class RecoveryTrainingDriver:
    """Collect one bounded recovery episode and commit its true continuation."""

    def __init__(self, wrapper, collector: RecoveryCollector, *, feature_summary: Callable,
                 base_policy: Callable, recovery_policy: Callable,
                 monitor: RecoveryMonitor | None = None, max_recovery_actions=20,
                 stable_cycles=3, handoff_at_cap=False,
                 handoff: Callable | None = None, on_event: Callable | None = None,
                 skill_name="single_recovery_skill"):
        if not callable(feature_summary) or not callable(base_policy) or not callable(recovery_policy):
            raise TypeError("feature_summary, base_policy and recovery_policy must be callable")
        if type(max_recovery_actions) is not int or max_recovery_actions < 1:
            raise ValueError("max_recovery_actions must be a positive integer")
        if type(stable_cycles) is not int or stable_cycles < 1:
            raise ValueError("stable_cycles must be a positive integer")
        if not isinstance(skill_name, str) or not skill_name:
            raise ValueError("skill_name must be a nonempty string")
        self.wrapper, self.collector = wrapper, collector
        self.feature_summary = feature_summary
        self.base_policy, self.recovery_policy = base_policy, recovery_policy
        self.monitor = monitor or RecoveryMonitor()
        self.max_recovery_actions = max_recovery_actions
        self.stable_cycles = stable_cycles
        self.handoff_at_cap = bool(handoff_at_cap)
        self.skill_name = skill_name
        self.handoff_callback, self.on_event = handoff, on_event

    @staticmethod
    def _terminal_info(info):
        execution = dict(info.get("execution_info", {}))
        if "reason" not in execution:
            execution["reason"] = "infrastructure_exception"
        execution.setdefault("valid_trial", False)
        execution.setdefault("incomplete", not bool(execution.get("terminated") or execution.get("truncated")))
        execution.setdefault("terminated", False)
        execution.setdefault("truncated", False)
        return execution

    def _emit(self, kind, **details):
        if self.on_event is not None:
            self.on_event(kind, details)

    def _base_step(self, observation):
        action = self.base_policy(observation)
        next_observation, reward, terminated, truncated, info = self.wrapper.step(action)
        return next_observation, reward, terminated, truncated, info

    def run(self, seed: int, condition=None, **reset_kwargs):
        observation, reset_info = self.wrapper.reset(seed, condition=condition, **reset_kwargs)
        self._emit("reset", seed=seed, observation_physics_step=observation.get("physics_step"),
                   reset_info=reset_info)
        recovery_actions = 0
        admitted = 0
        triggered = False
        selected_skill = None
        stable_count = 0
        terminal = {}
        while not self.wrapper.done:
            summary = dict(self.feature_summary(observation))
            if not triggered:
                decision = self.monitor.decide(summary)
                if decision.trigger:
                    triggered = True
                    selected_skill = (self.recovery_policy.select_skill(observation, summary)
                                      if hasattr(self.recovery_policy, "select_skill") else self.skill_name)
                    self._emit("recovery_trigger", reason=decision.reason, score=decision.score,
                               selected_skill=selected_skill, physics_step=observation.get("physics_step"))
                    if selected_skill is None:
                        if not self.wrapper.done:
                            self.wrapper.stop("no_reliable_skill")
                        terminal = {"reason": "no_reliable_skill", "valid_trial": False,
                                    "terminated": False, "truncated": True, "incomplete": True}
                        self.collector.discard()
                        return TrainingEpisodeResult("invalid", terminal["reason"], 0, 0, terminal)
            if triggered:
                if recovery_actions >= self.max_recovery_actions:
                    if not self.wrapper.done:
                        self.wrapper.stop("recovery_driver_budget")
                    self.collector.discard()
                    terminal = {"reason": "recovery_driver_budget", "valid_trial": False,
                                "terminated": False, "truncated": True, "incomplete": True}
                    self._emit("recovery_driver_budget", actions=recovery_actions)
                    return TrainingEpisodeResult("stopped", terminal["reason"], recovery_actions, admitted, terminal)
                action = self.recovery_policy(observation, summary, selected_skill)
                values = np.asarray(action, dtype=np.float32)
                result: CollectionResult = self.collector.step(observation, values)
                recovery_actions += 1
                observation = result.observation
                admitted += result.admitted
                if observation is None:
                    terminal = self._terminal_info(result.info)
                    return TrainingEpisodeResult("invalid", terminal["reason"], recovery_actions, admitted, terminal)
                terminal_info = self._terminal_info(result.info)
                if terminal_info.get("terminated") or terminal_info.get("truncated") or self.wrapper.done:
                    terminal = terminal_info
                    return TrainingEpisodeResult("terminated_during_recovery", terminal["reason"], recovery_actions, admitted, terminal)
                summary = dict(self.feature_summary(observation))
                if bool(summary.get("stable", False)):
                    stable_count += 1
                else:
                    stable_count = 0
                cap_reached = self.handoff_at_cap and recovery_actions >= self.max_recovery_actions
                if stable_count >= self.stable_cycles or cap_reached:
                    self.collector.handoff()
                    self._emit("handoff", actions=recovery_actions, stable_cycles=stable_count,
                               reason="action_cap" if cap_reached else "stable",
                               physics_step=observation.get("physics_step"))
                    if self.handoff_callback is not None:
                        self.handoff_callback(observation)
                    # The fixed base policy now owns the episode. Its final
                    # result is the only task label attached to the recovery
                    # fragment by RecoveryCollector.finish_continuation().
                    while not self.wrapper.done:
                        observation, _, terminated, truncated, info = self._base_step(observation)
                        if terminated or truncated or self.wrapper.done:
                            terminal = self._terminal_info(info)
                            admitted += self.collector.finish_continuation(terminal)
                            self._emit("continuation", outcome=terminal)
                            return TrainingEpisodeResult("committed", terminal["reason"], recovery_actions, admitted, terminal)
                    terminal = {"reason": "infrastructure_exception", "valid_trial": False,
                                "terminated": False, "truncated": True, "incomplete": True}
                    admitted += self.collector.finish_continuation(terminal)
                    return TrainingEpisodeResult("invalid", terminal["reason"], recovery_actions, admitted, terminal)
                continue
            observation, _, terminated, truncated, info = self._base_step(observation)
            if terminated or truncated or self.wrapper.done:
                terminal = self._terminal_info(info)
                return TrainingEpisodeResult("base_terminal", terminal["reason"], recovery_actions, admitted, terminal)
        terminal = {"reason": "wrapper_closed", "valid_trial": False, "terminated": False,
                    "truncated": True, "incomplete": True}
        self.collector.discard()
        return TrainingEpisodeResult("invalid", terminal["reason"], recovery_actions, admitted, terminal)
