"""Extract deployment-only recovery windows from recorded HDF5 episodes.

The extractor is intentionally separate from Isaac startup.  It uses only
persisted observations, actions and episode metadata, rejects mixed splits,
and emits windows for executed ``recovery_delta`` actions.  A dev/test file
cannot accidentally become a warm-start training parent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import ROOT
from evotac.data.rollout_logger import read_tree
from evotac.learning.recovery_observation import RECOVERY_OBSERVATION_DIM, RecoveryHistory
from evotac.policy.tactile_features import CachedTactileEncoder, FrozenTactileEncoder, recovery_features


def _metadata(handle):
    try:
        return json.loads(handle.attrs["scene_metadata"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("episode lacks valid scene_metadata") from exc


def extract(episodes, encoder, *, split="train", skill_name="small_lift_adjust_reapproach",
            task_budget_steps=1200, recovery_budget_steps=120, physical_hz=120):
    if not isinstance(skill_name, str) or not skill_name:
        raise ValueError("skill_name must be nonempty")
    rows, provenance = [], []
    seen_parents = set()
    for episode_path in episodes:
        with h5py.File(episode_path, "r") as handle:
            metadata = _metadata(handle)
            if metadata.get("split") != split:
                continue
            parent = metadata.get("parent_scene_id")
            if not isinstance(parent, str) or not parent:
                raise ValueError("episode parent_scene_id is required")
            actions = {int(index): read_tree(handle["actions"][index])
                       for index in handle["actions"]}
            transitions = {int(index): read_tree(handle["transitions"][index])
                           for index in handle["transitions"]}
            recovery_indices = [index for index, action in actions.items()
                                if action.get("proposed_action", {}).get("kind") == "recovery_delta"]
            if not recovery_indices:
                continue
            if parent in seen_parents:
                raise ValueError(f"duplicate parent in warm-start data: {parent}")
            seen_parents.add(parent)
            history = RecoveryHistory(length=8, task_budget_steps=task_budget_steps,
                                      recovery_budget_steps=recovery_budget_steps,
                                      physical_hz=physical_hz)
            for index in range(len(handle["observations"]) - 1):
                observation = read_tree(handle["observations"][str(index)])
                # The wrapper records the recovery budget in every
                # observation.  After handoff it intentionally freezes that
                # value while base continuation runs, so deriving it from
                # physics_step would manufacture a mismatch at the handoff
                # boundary and reject otherwise valid transitions.
                remaining = observation.get("remaining_recovery_steps")
                if type(remaining) is not int or not 0 <= remaining <= recovery_budget_steps:
                    raise ValueError("observation has invalid remaining_recovery_steps")
                features = recovery_features(observation, encoder, recovery_remaining_steps=remaining)
                window, mask = history.append(features)
                action = actions.get(index)
                transition = transitions.get(index, {})
                if (action is None or action.get("proposed_action", {}).get("kind") != "recovery_delta"
                        or action.get("status") != "executed" or action.get("physics_steps") != 6
                        or transition.get("valid_trial") is not True):
                    continue
                values = np.asarray(action["proposed_action"]["values"], dtype=np.float32)
                if values.shape != (7,) or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
                    raise ValueError(f"invalid recovery action in {episode_path}:{index}")
                rows.append((window.copy(), mask.copy(), values.copy(), parent))
            provenance.append({"split": split, "skill": skill_name,
                               "parent_scene_id": parent, "episode_path": str(episode_path),
                               "version": metadata.get("versions", {})})
    if not rows:
        raise ValueError(f"no executed recovery transitions found in split={split}")
    histories, masks, actions, parents = zip(*rows)
    return {
        "history": np.stack(histories).astype(np.float32),
        "mask": np.stack(masks).astype(bool),
        "action": np.stack(actions).astype(np.float32),
        "skill_indices": np.zeros(len(rows), dtype=np.int64),
        "metadata": {"schema": "evotac.recovery_history_dataset.v1",
                      "frame_dimension": RECOVERY_OBSERVATION_DIM,
                      "history_length": 8, "split": split, "skill_names": [skill_name],
                      "parents": sorted(set(parents)), "provenance": provenance},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=ROOT / "checkpoints/univtac_release/encoder.pth")
    parser.add_argument("--encoder-source", type=Path,
                        default=ROOT / "checkpoints/univtac_release/encoder_source.json")
    parser.add_argument("--split", choices=("train", "dev", "acceptance", "test"), default="train")
    parser.add_argument("--skill-name", default="small_lift_adjust_reapproach")
    args = parser.parse_args()
    provenance = json.loads(args.encoder_source.read_text())
    encoder = CachedTactileEncoder(FrozenTactileEncoder(
        args.encoder_checkpoint, provenance["sha256"], device="cpu"))
    result = extract(args.episodes, encoder, split=args.split, skill_name=args.skill_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, history=result["history"], mask=result["mask"],
                        action=result["action"], skill_indices=result["skill_indices"],
                        metadata=json.dumps(result["metadata"], allow_nan=False))
    print(json.dumps({"status": "written", "output": str(args.output),
                      "transitions": int(len(result["action"])),
                      "parents": len(result["metadata"]["provenance"]),
                      "split": args.split}, indent=2))


if __name__ == "__main__":
    main()
