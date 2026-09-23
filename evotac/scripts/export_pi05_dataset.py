"""Export train-only accepted control trajectories, with audited time alignment.

The student receives RGB/language/proprioception. Teachers may use tactile or
recovery; that provenance is explicit and is not a vision-only teacher claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.learning.recovery_experiment import sha256, scene_splits, resolved_split
from evotac.policy.pi05_adapter import encode_observation


def candidates(source_root, splits):
    selected, excluded = {}, []
    for path in sorted(source_root.glob("**/episodes/*.h5")):
        with h5py.File(path, "r") as handle:
            scene = json.loads(handle.attrs.get("scene_metadata", "{}"))
            if scene.get("split") != "train":
                continue  # Do not inspect held-out outcomes or observations.
            seed = scene.get("seed")
            parent = scene.get("parent_scene_id")
            if type(seed) is not int or parent != f"insert_hole:source_seed:{seed}" or resolved_split(seed, splits) != "train":
                raise ValueError(f"train parent registry mismatch: {path}")
            outcome = json.loads(handle.attrs.get("outcome", "{}"))
            if (not bool(handle.attrs.get("complete")) or outcome.get("incomplete") is not False
                    or outcome.get("reason") != "success" or outcome.get("valid_trial") is not True):
                excluded.append({"path": str(path), "reason": "not_complete_valid_train_success"})
                continue
            if parent in selected:
                excluded.append({"path": str(path), "reason": "duplicate_train_parent_first_lexical_selected"})
                continue
            selected[parent] = path
    return selected, excluded


def audited_rows(handle):
    """Reject gaps, unconfirmed commands, stale/invalid cameras and wrong joints."""
    keys = sorted(handle["actions"], key=int)
    if keys != list(map(str, range(len(keys)))) or len(handle["observations"]) != len(keys) + 1:
        raise ValueError("non-contiguous trajectory")
    previous_end = None
    for index, key in enumerate(keys):
        execution = read_tree(handle["actions"][key])
        start, end, steps = (execution[k] for k in ("physics_start", "physics_end", "physics_steps"))
        if previous_end is not None and start != previous_end:
            raise ValueError("physics gap")
        previous_end = end
        if steps != 6:
            if index == len(keys) - 1 and 0 < steps < 6:
                break  # Only the terminal partial action can be excluded.
            raise ValueError("non-20Hz action inside trajectory")
        if (end - start != 6 or execution["status"] != "executed" or execution["valid_trial"] is not True
                or not np.isclose(execution["sim_time_start"], start / 120, rtol=0, atol=1e-8)
                or not np.isclose(execution["sim_time_end"], end / 120, rtol=0, atol=1e-8)):
            raise ValueError("execution timing/status mismatch")
        applied = execution["applied_action"]
        if any(applied.get(k) is not True for k in
               ("accepted", "accepted_by_target_api", "simulation_submission_confirmed")):
            raise ValueError("unconfirmed target submission")
        command = execution["commanded_action"]
        if command["kind"] not in {"joint_target", "recovery_delta"} or command["force"] is not False:
            raise ValueError("unsupported action contract")
        target = np.r_[np.asarray(applied["arm_position"]), np.mean(applied["finger_position"])].astype(np.float32)
        expected = np.r_[command["joint_target"], command["gripper_target"]]
        if target.shape != (8,) or not np.isfinite(target).all() or not np.allclose(target, expected, rtol=0, atol=1e-6):
            raise ValueError("accepted and commanded targets disagree")
        obs = handle["observations"][key]
        if int(obs["physics_step"][()]) != start or int(handle["observations"][str(index + 1)]["physics_step"][()]) != end:
            raise ValueError("observation/action alignment mismatch")
        convention = read_tree(obs["proprioception_convention"])
        if (list(convention["arm_joint_names"]) != [f"panda_joint{i}" for i in range(1, 8)]
                or list(convention["finger_joint_names"]) != ["panda_finger_joint1", "panda_finger_joint2"]
                or convention["arm_position_unit"] != "rad" or convention["finger_position_unit"] != "m"):
            raise ValueError("unconfirmed joint order/units")
        # Whitelist HDF5 reads: no tactile or privileged-state tree is loaded.
        observation = {"proprioception": read_tree(obs["proprioception"]),
                       "images": {"camera": read_tree(obs["images"]["camera"])}}
        for camera in ("head", "wrist"):
            rgb = observation["images"]["camera"][camera]["rgb"]
            if rgb["sample_physics_step"] != start or rgb["age_steps"] != 0:
                raise ValueError("stale camera at action boundary")
        inputs, _ = encode_observation(observation)
        yield inputs, target, start, command["kind"]


def export(source_root, destination, splits):
    selected, excluded = candidates(source_root, splits)
    if not selected:
        raise ValueError("no eligible train rollouts")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "evotac.pi05_dataset.v2", "status": "building", "source_split": "train",
                "tactile_inputs": False, "teacher": "FTP1 with optional SAC/scripted recovery",
                "selection": "complete valid train success; first lexical path per parent",
                "control_hz": 20, "physical_hz": 120, "action_representation": "absolute",
                "state": "panda_joint1..7 rad; mean finger displacement m",
                "episodes": [], "excluded": excluded, "source_root": str(source_root)}
    states, actions = [], []
    try:
        for index, (parent, path) in enumerate(sorted(selected.items())):
            before = sha256(path)
            with h5py.File(path, "r") as handle:
                scene = json.loads(handle.attrs["scene_metadata"])
                rows = list(audited_rows(handle))
            if not rows or sha256(path) != before:
                raise ValueError(f"empty or changing source: {path}")
            arrays = {name: np.stack([row[0][name] for row in rows]) for name in rows[0][0]}
            arrays.update(action=np.stack([row[1] for row in rows]), physics_step=np.array([row[2] for row in rows]))
            episode = destination / f"episode_{index:04d}.h5"
            with h5py.File(episode, "x") as out:
                for name, value in arrays.items():
                    out.create_dataset(name, data=value, compression="lzf")
            states.append(arrays["observation.state"])
            actions.append(arrays["action"])
            manifest["episodes"].append({"file": episode.name, "sha256": sha256(episode), "frames": len(rows),
                "parent_scene_id": parent, "source_seed": scene["seed"], "split": "train",
                "source": str(path), "source_sha256": before, "source_versions": scene.get("versions"),
                "teacher_action_kinds": sorted({row[3] for row in rows})})
        stats = {}
        for name, batches in (("observation.state", states), ("action", actions)):
            array = np.concatenate(batches)
            stats[name] = {"mean": array.mean(0).tolist(), "std": array.std(0).tolist(),
                           "q01": np.quantile(array, .01, axis=0).tolist(),
                           "q99": np.quantile(array, .99, axis=0).tolist()}
        (destination / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        manifest.update(status="completed", stats_sha256=sha256(destination / "stats.json"),
                        episode_count=len(states), frame_count=sum(len(state) for state in states))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "datasets/phase1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1_insert_hole.yaml")
    parser.add_argument("--output", type=Path, required=True, help="absolute or relative to evotac/")
    args = parser.parse_args()
    manifest = export(args.source_root, output_path(args.output), scene_splits(args.config))
    print(json.dumps({k: manifest[k] for k in ("status", "episode_count", "frame_count")}))


if __name__ == "__main__":
    main()
