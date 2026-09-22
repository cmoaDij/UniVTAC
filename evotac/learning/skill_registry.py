"""Explicit skill metadata; only one recovery skill is enabled initially."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class SkillSpec:
    name: str
    action_dim: int
    description: str
    enabled: bool = True
    applicability: tuple[str, ...] = field(default_factory=tuple)
    policy_factory: Callable | None = None


class SkillRegistry:
    def __init__(self, skills=()):
        self._skills = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: SkillSpec):
        if not isinstance(skill, SkillSpec) or not skill.name or skill.action_dim < 1:
            raise ValueError("invalid skill specification")
        if skill.name in self._skills:
            raise ValueError(f"skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name):
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc
        if not skill.enabled:
            raise ValueError(f"skill is disabled: {name}")
        return skill

    def names(self):
        return tuple(self._skills)

    def manifest(self):
        return [{"name": skill.name, "action_dim": skill.action_dim,
                 "description": skill.description, "enabled": skill.enabled,
                 "applicability": list(skill.applicability)} for skill in self._skills.values()]


def initial_recovery_registry():
    """Return the conservative single-skill registry used by P3 smoke runs."""
    return SkillRegistry([SkillSpec(
        name="small_lift_adjust_reapproach", action_dim=7,
        description="small lift, lateral adjustment and re-approach recovery",
        applicability=("object_lost_risk", "contact_blocked", "budget_remaining"),
    )])


def planned_recovery_registry():
    """Return the three frozen skill slots planned for the P3/P4 handoff.

    The registry only describes action-compatible policy slots. It does not
    claim that any slot has passed a real Isaac/TacEx effectiveness test.
    """
    return SkillRegistry([
        SkillSpec(
            name="small_lift_adjust_reapproach", action_dim=7,
            description="small lift, lateral adjustment and re-approach recovery",
            applicability=("object_lost_risk", "budget_remaining"),
        ),
        SkillSpec(
            name="lateral_align", action_dim=7,
            description="lateral contact alignment before re-approach",
            applicability=("contact_blocked", "lateral_offset"),
        ),
        SkillSpec(
            name="orientation_adjust", action_dim=7,
            description="orientation correction while preserving the grasp",
            applicability=("contact_blocked", "angle_offset"),
        ),
    ])
