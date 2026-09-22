"""P2 physical paired continuations with a shared total task budget.

Fresh sources and both replays can run under one immutable runtime manifest.
No-recovery preserves the pending FTP-1 chunk; scripted recovery clears it.
Both branches start the same declared model RNG stream for future inference.
This is a development experiment, not a trained-skill effectiveness label.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import yaml

from evotac.config import ROOT, load_config
from evotac.data.rollout_logger import read_tree, write_tree
from evotac.data.schemas import Action, json_value
from evotac.envs.state_replay import validate_scene
from evotac.policy.ftp1_adapter import FTP1Policy, decode_chunk, encode_observation
from evotac.policy.ftp1_client import FTP1Client
from evotac.scripts.replay_chunk16_prefix import build_scene, hardware_comparison
from evotac.scripts.runtime import build_wrapper, launch, parser_for


def _outcome(wrapper):
    if wrapper is None or wrapper.logger is None:
        return None
    raw = wrapper.logger.handle.attrs.get("outcome")
    return json.loads(raw) if raw else None


def saved_pending(source_run_id, checks, prefix_controls):
    """Reconstruct unexecuted targets from the durable inference chunk."""
    controls = checks["controls"][:prefix_controls]
    request = controls[-1]["inference"]["request_id"]
    start = next(i for i, row in enumerate(controls)
                 if row["inference"]["request_id"] == request)
    with h5py.File(checks["episode_path"], "r") as handle:
        _, qpos = encode_observation(read_tree(handle["observations"][str(start)]))
    model = checks["model"]
    cfg = model["config"]
    chunk = np.load(ROOT / "runs/phase1" / source_run_id / "policy" /
                    f"chunk_{request:05d}.npy", allow_pickle=False)
    actions = decode_chunk(chunk, qpos, model["action_joint_rep"],
                           cfg["chunk_first_index"], cfg["execute_chunk_steps"], cfg["chunk_stride"])
    return [action.as_dict() for action in actions[prefix_controls-start:]]


def run_branch(wrapper, policy, scene, prompt, *, branch, pending_actions,
               recovery_action, recovery_controls, post_prefix_controls,
               inference_seed, progress=None):
    if branch not in {"no_recovery", "scripted_recovery"}:
        raise ValueError("Unknown branch")
    policy.reset()
    observation, reconstruction = wrapper.replay_start(scene)
    row = {"branch": branch, "status": "replaying", "reconstruction": reconstruction,
           "episode_path": str(wrapper.logger.path), "recovery_controls": [],
           "continuation_controls": [], "post_prefix_limit": post_prefix_controls}
    if not reconstruction.get("valid_match") or not reconstruction.get("versions_match"):
        wrapper.stop("counterfactual_replay_rejected")
        row.update(status="replay_rejected", outcome=_outcome(wrapper))
        return row
    row["inference_rng"] = policy.backend.reset_seed(inference_seed)
    row["source_pending_actions"] = len(pending_actions)
    if branch == "no_recovery":
        policy.pending.extend(Action(**action) for action in pending_actions)
    row["restored_pending_actions"] = len(policy.pending)
    wrapper.logger.event("paired_branch_start", {
        "branch": branch, "scene_sha256": scene["integrity_sha256"],
        "inference_seed": inference_seed, "post_prefix_limit": post_prefix_controls,
        "source_pending_actions": len(pending_actions),
        "restored_pending_actions": len(policy.pending)})
    start_elapsed = wrapper.task_elapsed
    start = time.perf_counter()
    try:
        for control in range(post_prefix_controls):
            if wrapper.done:
                break
            recovery = branch == "scripted_recovery" and control < recovery_controls
            if recovery:
                action = Action("recovery_delta", recovery_action)
            else:
                if branch == "scripted_recovery" and control == recovery_controls:
                    policy.switch_control()
                    wrapper.logger.event("scripted_handoff", {"recovery_controls": recovery_controls,
                                                               "physics_step": observation["physics_step"]})
                action = policy.next_action(observation, prompt)
            observation, _, terminated, truncated, info = wrapper.step(action)
            row["recovery_controls" if recovery else "continuation_controls"].append(info["execution_info"])
            if progress and (control % 16 == 0 or terminated or truncated):
                progress(branch, control+1, info["execution_info"].get("reason"))
            if terminated or truncated:
                break
        if not wrapper.done:
            wrapper.stop("counterfactual_control_limit")
        row["status"] = "finished"
    except Exception as exc:
        wrapper.stop("counterfactual_exception")
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    row.update(outcome=_outcome(wrapper),
               post_prefix_physics_steps=wrapper.task_elapsed-start_elapsed,
               wall_seconds=time.perf_counter()-start)
    return row


def main():
    parser = parser_for(__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run-id")
    source.add_argument("--fresh-source", action="store_true",
                        help="Generate and seal the source before replaying both branches")
    parser.add_argument("--policy-config", type=Path, default=ROOT / "configs/ftp1_insert_hole_chunk16.yaml")
    parser.add_argument("--model-gpu", default="8")
    parser.add_argument("--prefix-controls", type=int, default=12)
    parser.add_argument("--recovery-controls", type=int, default=4)
    parser.add_argument("--post-prefix-controls", type=int,
                        help="Shared cap including recovery; default uses the full remaining task budget")
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--recovery-action", nargs=7, type=float,
                        default=[0, 0, 0.5, 0, 0, 0, -0.5])
    args = parser.parse_args()
    config = load_config(args.config)
    remaining = config["budgets"]["task_physics_steps"] // 6 - args.prefix_controls
    controls = remaining if args.post_prefix_controls is None else args.post_prefix_controls
    if (args.prefix_controls < 1 or remaining < 1 or not 1 <= controls <= remaining
            or not 1 <= args.recovery_controls < controls
            or args.recovery_controls*6 >= config["budgets"]["recovery_physics_steps"]):
        raise ValueError("Control counts must fit the task/recovery budgets and allow a handoff")
    if (not np.isfinite(args.recovery_action).all() or np.max(np.abs(args.recovery_action)) > 1
            or not 0 <= args.inference_seed < 2**32):
        raise ValueError("Invalid recovery action or inference seed")
    policy_config = yaml.safe_load(args.policy_config.read_text())
    policy_config.update(model_gpu=args.model_gpu, inference_seed=args.seed)
    config, paths, versions, launcher = launch(args)
    wrapper = backend = None
    result = {"status": "starting", "protocol": "evotac.paired_continuation.v2",
              "evidence_scope": "development scripted recovery; not trained-skill effectiveness",
              "source_run_id": args.source_run_id, "fresh_source": args.fresh_source,
              "seed": args.seed, "prefix_controls": args.prefix_controls,
              "recovery_controls": args.recovery_controls, "recovery_action": args.recovery_action,
              "post_prefix_controls": controls, "task_budget_steps": config["budgets"]["task_physics_steps"],
              "inference_seed": args.inference_seed, "paired_valid": False, "branches": []}

    def save():
        target = paths["run_root"] / "checks.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(json_value(result), indent=2, allow_nan=False))
        temporary.replace(target)

    def progress(branch, count, reason):
        result["progress"] = {"branch": branch, "controls": count, "reason": reason}
        save()
        print(f"{branch}: {count}/{controls} post-prefix controls; {reason}", flush=True)

    save()
    try:
        if not args.fresh_source:
            scene, episode, source_checks = build_scene(args.source_run_id, args.prefix_controls)
            validate_scene(scene, versions, {})
            if scene["seed"] != args.seed:
                raise ValueError("Seed does not match the immutable source")
            manifest = json.loads((paths["dataset_root"] / "manifest.json").read_text())
            hardware = hardware_comparison(args.source_run_id, manifest)
            if not hardware.get("same_physical_device") or not hardware.get("same_launch_mode"):
                raise ValueError("Source and replay must use the same GPU and launch mode")
            source_cfg = source_checks["model"]["config"]
            if any(source_cfg.get(key) != value for key, value in policy_config.items()
                   if key not in {"model_gpu", "inference_seed"}):
                raise ValueError("Source policy configuration differs")
            pending = saved_pending(args.source_run_id, source_checks, args.prefix_controls)
            result.update(source_episode_path=str(episode), hardware_comparison=hardware)
        backend = FTP1Client(policy_config, paths["run_root"])
        policy = FTP1Policy(backend, policy_config)
        result["model"] = backend.metadata
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        if args.fresh_source:
            result["status"] = "building_source"
            save()
            observation, _ = wrapper.reset(args.seed)
            for _ in range(args.prefix_controls):
                observation, _, _, _, _ = wrapper.step(policy.next_action(observation, policy_config["prompt"]))
                if wrapper.done:
                    result.update(status="source_terminal_before_boundary", source_outcome=_outcome(wrapper),
                                  source_episode_path=str(wrapper.logger.path))
                    return
            scene = wrapper.export_scene()
            pending = [action.as_dict() for action in policy.pending]
            result["source_episode_path"] = str(wrapper.logger.path)
            # Keep this wrapper live.  replay_start will close the source
            # episode and reset the same task instance for each paired branch.
        with h5py.File(paths["run_root"] / "source_scene.h5", "x") as handle:
            write_tree(handle, scene)
        result.update(status="running", scene_sha256=scene["integrity_sha256"],
                      parent_scene_id=scene["parent_scene_id"], split=scene["split"],
                      source_pending_actions=pending, versions=versions)
        save()
        for branch in ("no_recovery", "scripted_recovery"):
            try:
                row = run_branch(wrapper, policy, scene, policy_config["prompt"], branch=branch,
                                 pending_actions=pending, recovery_action=args.recovery_action,
                                 recovery_controls=args.recovery_controls,
                                 post_prefix_controls=controls, inference_seed=args.inference_seed,
                                 progress=progress)
            except Exception as exc:
                wrapper.stop("counterfactual_exception")
                row = {"branch": branch, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "outcome": _outcome(wrapper), "episode_path": str(wrapper.logger.path)}
            result["branches"].append(row)
            save()
        result["paired_valid"] = all(
            row.get("status") == "finished" and row["reconstruction"].get("valid_match") is True
            and row["outcome"].get("valid_trial") is True
            and row["outcome"].get("incomplete") is False
            and row["outcome"].get("reason") in {"success", "object_lost", "task_budget", "execution_rejected"}
            for row in result["branches"])
        result["status"] = "completed" if result["paired_valid"] else "completed_with_ineligible_branches"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        if wrapper is not None:
            wrapper.stop("counterfactual_runner_closed")
        save()
        if backend is not None:
            backend.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
