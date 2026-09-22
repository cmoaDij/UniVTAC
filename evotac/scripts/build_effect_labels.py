"""Build versioned P4 effect labels from frozen-skill HDF5 episodes.

Only complete acceptance/test episodes with an explicit violation metric are
emitted.  ``object_lost`` is a task outcome and is deliberately not inferred
as a constraint violation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from evotac.config import ROOT
from evotac.data.rollout_logger import read_tree
from evotac.learning.effect_model import EffectHistoryEncoder
from evotac.learning.recovery_observation import RecoveryHistory
from evotac.policy.tactile_features import CachedTactileEncoder, FrozenTactileEncoder, recovery_features


def build(episodes, tactile_encoder, effect_encoder, *, split, effect_version,
          skill_name, task_budget_steps=1200, recovery_budget_steps=120, physical_hz=120):
    labels = defaultdict(dict)
    skipped = defaultdict(int)
    for episode_path in episodes:
        with h5py.File(episode_path, "r") as handle:
            try:
                metadata = json.loads(handle.attrs["scene_metadata"])
                outcome = json.loads(handle.attrs["outcome"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"episode metadata/outcome is invalid: {episode_path}") from exc
            if metadata.get("split") != split:
                continue
            parent = metadata.get("parent_scene_id")
            if not isinstance(parent, str) or not parent:
                raise ValueError("effect label episode needs parent_scene_id")
            if outcome.get("valid_trial") is not True or outcome.get("incomplete") is True:
                skipped["invalid_or_incomplete"] += 1
                continue
            reason = outcome.get("reason")
            if reason not in {"success", "object_lost", "task_budget"}:
                skipped["unsupported_outcome"] += 1
                continue
            if "violation" not in outcome:
                skipped["missing_explicit_violation"] += 1
                continue
            violation = float(outcome["violation"])
            if not np.isfinite(violation) or violation < 0 or violation > 1:
                raise ValueError("violation must be an explicit probability in [0, 1]")
            actions = {int(index): read_tree(handle["actions"][index]) for index in handle["actions"]}
            recovery_indices = [index for index, action in actions.items()
                                if action.get("proposed_action", {}).get("kind") == "recovery_delta"]
            if not recovery_indices:
                skipped["no_recovery_segment"] += 1
                continue
            if parent in labels and skill_name in labels[parent]:
                raise ValueError(f"duplicate parent/skill label: {parent}/{skill_name}")
            first_recovery = min(recovery_indices)
            recovery_start = int(actions[first_recovery]["physics_start"])
            history = RecoveryHistory(length=8, task_budget_steps=task_budget_steps,
                                      recovery_budget_steps=recovery_budget_steps,
                                      physical_hz=physical_hz)
            first_window = first_mask = None
            recovery_cost = 0
            for index in range(len(handle["observations"]) - 1):
                observation = read_tree(handle["observations"][str(index)])
                remaining = max(0, recovery_budget_steps -
                                max(0, int(observation["physics_step"]) - recovery_start))
                features = recovery_features(observation, tactile_encoder, recovery_remaining_steps=remaining)
                window, mask = history.append(features)
                if index in recovery_indices:
                    action = actions[index]
                    if action.get("status") != "executed" or action.get("physics_steps") != 6:
                        raise ValueError(f"invalid recovery action in {episode_path}:{index}")
                    recovery_cost += int(action["physics_steps"])
                if index == first_recovery:
                    first_window, first_mask = window.copy(), mask.copy()
            if first_window is None:
                raise ValueError("recovery segment has no observation window")
            with torch.inference_mode():
                embedding = effect_encoder(first_window[None], first_mask[None]).squeeze(0).cpu().numpy()
            labels[parent][skill_name] = {
                "embedding": embedding.tolist(), "task_success": float(reason == "success"),
                "recovery_cost": min(1.0, recovery_cost / task_budget_steps),
                "violation": violation, "skill_name": skill_name,
                "version": effect_version, "valid": True,
            }
    if not labels:
        raise ValueError(f"no valid effect labels; skipped={dict(skipped)}")
    rows = [{"parent_scene_id": parent, "split": split, "effect_version": effect_version,
             "embedding": next(iter(skills.values()))["embedding"], "skills": skills,
             "provenance": {"episode_count": len(skills)}} for parent, skills in labels.items()]
    return rows, dict(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--effect-checkpoint", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=ROOT / "checkpoints/univtac_release/encoder.pth")
    parser.add_argument("--encoder-source", type=Path,
                        default=ROOT / "checkpoints/univtac_release/encoder_source.json")
    parser.add_argument("--split", choices=("acceptance", "test"), required=True)
    parser.add_argument("--effect-version", required=True)
    parser.add_argument("--skill-name", required=True)
    args = parser.parse_args()
    source = json.loads(args.encoder_source.read_text())
    tactile = CachedTactileEncoder(FrozenTactileEncoder(args.encoder_checkpoint, source["sha256"], device="cpu"))
    effect_state = torch.load(args.effect_checkpoint, map_location="cpu", weights_only=False)
    effect_encoder = EffectHistoryEncoder.from_checkpoint(effect_state, device="cpu").eval()
    rows, skipped = build(args.episodes, tactile, effect_encoder, split=args.split,
                          effect_version=args.effect_version, skill_name=args.skill_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"status": "written", "rows": len(rows), "output": str(args.output),
                      "split": args.split, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
