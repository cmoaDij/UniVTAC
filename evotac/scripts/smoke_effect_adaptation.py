"""CPU smoke test for P4/P5 effect selection and candidate acceptance."""
import json

import numpy as np

from evotac.learning.continual_adaptation import AcceptanceGate, CandidateEvaluation, adaptation_mix
from evotac.learning.skill_selector import EffectMemory, EffectRecord, SkillSelector


def main():
    memory = EffectMemory()
    memory.add(EffectRecord(np.zeros(4), "small_lift_adjust_reapproach", 1.0, 4.0, 0.0, "v1"))
    nearest = memory.retrieve(np.ones(4) * 0.01, k=1, version="v1")
    selection = SkillSelector(["small_lift_adjust_reapproach"]).select({
        "small_lift_adjust_reapproach": {"success_probability": 0.9, "cost": 2.0, "violation_probability": 0.0}})
    baseline = CandidateEvaluation("base", (0.2, 0.3, 0.4), (0.8, 0.9, 0.8))
    candidate = CandidateEvaluation("candidate", (0.5, 0.6, 0.7), (0.8, 0.9, 0.8))
    decision = AcceptanceGate().evaluate(baseline, candidate)
    mixed = adaptation_mix([1, 2], [3, 4], [5, 6], seed=2)
    print(json.dumps({"status": "ok", "nearest": len(nearest), "selected": selection.skill_name,
                      "accepted": decision.accepted, "mix_size": len(mixed)}, indent=2))


if __name__ == "__main__":
    main()
