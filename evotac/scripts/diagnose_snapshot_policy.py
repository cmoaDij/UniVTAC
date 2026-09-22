"""Offline frozen-policy sensitivity on aligned dev snapshot observations.

This is an inference diagnostic, never a rollout, efficacy test, or audit gate.
Each recovery query starts a fresh history at its hypothetical entry boundary.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch
import yaml

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.envs.state_replay import scene_digest
from evotac.learning.recovery_experiment import sha256
from evotac.learning.recovery_observation import RecoveryHistory, recovery_summary
from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_sac import RecoverySAC
from evotac.policy.ftp1_adapter import encode_observation, decode_chunk
from evotac.policy.ftp1_client import FTP1Client
from evotac.policy.tactile_features import CachedTactileEncoder, FrozenTactileEncoder, recovery_features
from evotac.scripts.run_recovery_paired_eval import _load_actor


def validate_pair(source_meta, restored_meta, source, restored):
    if source_meta.get("split") != "dev" or restored_meta.get("split") != "dev":
        raise ValueError("Only dev observations are eligible for sensitivity diagnosis")
    for key in ("parent_scene_id", "seed", "condition", "versions"):
        if key not in source_meta or source_meta[key] != restored_meta.get(key):
            raise ValueError(f"Source/restoration metadata mismatch: {key}")
    for key in ("physics_step", "remaining_physics_steps", "remaining_recovery_steps", "sim_time"):
        if source[key] != restored[key]:
            raise ValueError(f"Unaligned observations: {key}")


def observation_variants(source, restored):
    result = {"source": copy.deepcopy(source), "source_repeat": copy.deepcopy(source),
              "restored": copy.deepcopy(restored)}
    for modality in ("camera", "tactile", "proprioception"):
        hybrid = copy.deepcopy(source)
        if modality == "proprioception":
            hybrid[modality] = copy.deepcopy(restored[modality])
        else:
            hybrid["images"][modality] = copy.deepcopy(restored["images"][modality])
        result[f"{modality}_only"] = hybrid
    return result


def action_difference(reference, candidate, groups):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    width = sum(stop - start for start, stop in groups.values())
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != width or not a.size
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Expected matching finite action matrices")
    delta = b - a
    return {name: {"max_abs": float(np.abs(delta[:, start:stop]).max()),
                   "rmse": float(np.sqrt(np.mean(delta[:, start:stop] ** 2))),
                   "per_dimension_max_abs": np.abs(delta[:, start:stop]).max(0).tolist()}
            for name, (start, stop) in groups.items()}


def load_pairs(run, branch):
    checks = json.loads((run / "checks.json").read_text())
    source_path = Path(checks["source_episode_path"])
    manifest_path = source_path.parent.parent / "manifest.json"
    matches = []
    for path in sorted(source_path.parent.glob("*.h5")):
        with h5py.File(path) as handle:
            meta = json.loads(handle.attrs["scene_metadata"])
            if meta.get("branch_id") == branch:
                matches.append(path)
    if len(matches) != 1:
        raise ValueError("Expected exactly one declared restored probe branch")
    scene_path = run / "source_scene.h5"
    pairs = []
    with h5py.File(source_path) as source, h5py.File(matches[0]) as restored, h5py.File(scene_path) as scene:
        scene_data = read_tree(scene)
        scene_hash = scene_data.pop("integrity_sha256")
        if (scene_data.get("integrity_algorithm") != "evotac.scene_binary.v1"
                or scene_digest(scene_data) != scene_hash or scene_hash != checks["source_scene_sha256"]):
            raise ValueError("Source scene integrity mismatch")
        source_meta = json.loads(source.attrs["scene_metadata"])
        restored_meta = json.loads(restored.attrs["scene_metadata"])
        if not source.attrs["complete"] or not restored.attrs["complete"]:
            raise ValueError("Source files are still being written")
        for step in range(checks["probe_controls"] + 1):
            # The scene observation includes the source's snapshot-time render;
            # the episode's prefix-boundary observation predates that render.
            original = read_tree(scene["observation"]) if step == 0 else read_tree(
                source["observations"][str(checks["prefix_controls"] + step)])
            candidate = read_tree(restored["observations"][str(step)])
            validate_pair(source_meta, restored_meta, original, candidate)
            pairs.append((original, candidate))
    files = [run / "checks.json", scene_path, source_path, matches[0], manifest_path]
    return checks, json.loads(manifest_path.read_text())["config"], pairs, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="probe_candidate", choices=[
        "probe_candidate", "probe_control_a", "probe_control_b", "probe_control_c", "probe_control_d"])
    parser.add_argument("--model-gpu", required=True)
    parser.add_argument("--inference-seed", type=int, default=23)
    args = parser.parse_args()
    args.output = output_path(args.output.resolve())
    checks, config, pairs, inputs = load_pairs(args.source_run.resolve(), args.branch)
    checkpoint, warmstart = Path(checks["trainer_checkpoint"]), Path(checks["warmstart"])
    for path, expected in ((checkpoint, checks["checkpoint_sha256"]), (warmstart, checks["warmstart_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Frozen weight hash mismatch: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.inference_seed)
    learner = RecoverySAC(128, 7, hidden_dim=128, device="cpu", seed=0)
    _load_actor(checkpoint, warmstart, learner)
    warm_state = torch.load(warmstart, map_location="cpu", weights_only=False)
    if warm_state.get("schema") != "evotac.control_warmstart.v1":
        raise ValueError("Warmstart schema mismatch")
    history_encoder = HistoryEncoder.from_checkpoint(warm_state["encoder"], device="cpu", frozen=True)
    history_encoder.eval()
    encoder_provenance = ROOT / "checkpoints/univtac_release/encoder_source.json"
    encoder_path = ROOT / "checkpoints/univtac_release/encoder.pth"
    encoder = CachedTactileEncoder(FrozenTactileEncoder(
        encoder_path, json.loads(encoder_provenance.read_text())["sha256"], device="cpu"))
    policy_path = ROOT / "configs/ftp1_insert_hole_chunk16.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    policy.update(model_gpu=args.model_gpu, inference_seed=args.inference_seed)
    inputs.extend([checkpoint, warmstart, encoder_provenance, encoder_path, policy_path])
    code = [*sorted((ROOT / "policy").glob("*.py")), *sorted((ROOT / "learning").glob("*.py")),
            Path(__file__), ROOT / "scripts/ftp1_worker.py", ROOT / "scripts/run_recovery_paired_eval.py"]
    report = {"schema": "evotac.snapshot_policy_sensitivity.v1", "status": "running", "split": "dev",
              "scope": "offline inference only; no executed actions, rollout, efficacy or audit certification",
              "recovery_history": "fresh length-8 masked history at each hypothetical entry boundary",
              "ftp1_history": "fresh inference at each boundary; queued deployment actions not executed",
              "inference_seed": args.inference_seed, "branch": args.branch,
              "strict_audit_passed": False, "test_started": False,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "inputs_sha256": {str(p.resolve()): sha256(p) for p in inputs},
              "code_sha256": {str(p.resolve()): sha256(p) for p in code}, "boundaries": []}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    arrays = {}
    groups = {"ftp1": {"arm_rad": (0, 7), "finger_m": (7, 8)},
              "recovery": {"normalized_translation": (0, 3), "normalized_rotation": (3, 6),
                           "normalized_gripper": (6, 7)}}
    backend = None
    try:
        backend = FTP1Client(policy, args.output)
        for step, (source, restored) in enumerate(pairs):
            outputs, summaries, requests = {}, {}, {}
            for variant, observation in observation_variants(source, restored).items():
                backend.reset_seed(args.inference_seed)
                model_inputs, base = encode_observation(observation)
                chunk, metadata = backend.infer(model_inputs, policy["prompt"])
                actions = decode_chunk(chunk, base, backend.metadata["action_joint_rep"],
                                       policy["chunk_first_index"], policy["execute_chunk_steps"], policy["chunk_stride"])
                ftp1 = np.stack([action.values for action in actions])
                features = recovery_features(observation, encoder,
                    recovery_remaining_steps=int(observation["remaining_recovery_steps"]))
                history = RecoveryHistory(length=8, physical_hz=int(config["simulation"]["physical_hz"]),
                    task_budget_steps=int(config["budgets"]["task_physics_steps"]),
                    recovery_budget_steps=int(config["budgets"]["recovery_physics_steps"]))
                frames, mask = history.append(features)
                with torch.inference_mode():
                    state = history_encoder(frames[None], mask[None]).squeeze(0).cpu().numpy()
                outputs[variant] = {"ftp1": ftp1, "recovery": learner.act(state, deterministic=True)[None]}
                summaries[variant] = recovery_summary(features)
                requests[variant] = metadata["request_id"]
                for name, value in outputs[variant].items():
                    arrays[f"step{step}_{variant}_{name}"] = value
            comparisons = {variant: {name: action_difference(outputs["source"][name], values[name], group)
                                    for name, group in groups.items()}
                           for variant, values in outputs.items() if variant != "source"}
            report["boundaries"].append({"probe_step": step, "physics_step": source["physics_step"],
                "comparisons_to_source": comparisons, "monitor_summaries": summaries,
                "ftp1_request_ids": requests})
            report_path.write_text(json.dumps(report, indent=2))
            print(f"completed aligned boundary {step}/{len(pairs)-1}", flush=True)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if backend is not None:
            backend.close()
        action_path = args.output / "actions.npz"
        np.savez_compressed(action_path, **arrays)
        report["actions_sha256"] = sha256(action_path)
        report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
