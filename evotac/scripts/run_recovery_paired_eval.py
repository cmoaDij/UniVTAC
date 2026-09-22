"""Evaluate FTP-1 and a frozen recovery actor from one immutable source prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from evotac.config import ROOT
from evotac.data.rollout_logger import write_tree
from evotac.data.schemas import Action, json_value
from evotac.evaluation.snapshot_audit import observation_audit
from evotac.learning.recovery_experiment import sha256
from evotac.learning.recovery_observation import RecoveryHistory, recovery_summary
from evotac.learning.recovery_training import RecoveryTrainingDriver
from evotac.learning.recovery_experiment import evaluate_episode
from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_sac import RecoverySAC
from evotac.policy.ftp1_adapter import FTP1Policy
from evotac.policy.ftp1_client import FTP1Client
from evotac.policy.tactile_features import CachedTactileEncoder, FrozenTactileEncoder, recovery_features
from evotac.scripts.runtime import build_wrapper, launch, parser_for


def _outcome(wrapper):
    raw = wrapper.logger.handle.attrs.get("outcome") if wrapper.logger else None
    return json.loads(raw) if raw else None


def _load_actor(checkpoint, warmstart, learner):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != "evotac.recovery_sac_trainer.v1":
        raise ValueError("unsupported trainer checkpoint")
    contract = state.get("experiment_contract", {})
    if contract.get("warmstart_sha256") != sha256(warmstart):
        raise ValueError("trainer checkpoint warm-start provenance mismatch")
    learner.load_state_dict(state["learner"])
    learner.actor.eval()
    return state


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--policy-config", type=Path, default=ROOT / "configs/ftp1_insert_hole_chunk16.yaml")
    parser.add_argument("--model-gpu", default="8")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    # parser_for already provides --seed; this evaluator requires it below.
    parser.add_argument("--prefix-controls", type=int, default=12)
    parser.add_argument("--post-prefix-controls", type=int, default=188)
    parser.add_argument("--recovery-controls", type=int, default=19)
    parser.add_argument("--monitor-object-lost-risk", type=float, default=0.3,
                        help="recorded for provenance; strict paired branch starts recovery explicitly")
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--trainer-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0 or args.prefix_controls < 1 or args.post_prefix_controls < 1:
        raise ValueError("seed and control counts must be positive")
    config, paths, versions, launcher = launch(args)
    policy_cfg = yaml.safe_load(args.policy_config.read_text())
    policy_cfg.update(model_gpu=args.model_gpu, inference_seed=args.seed)
    backend = wrapper = None
    report = {"status": "starting", "protocol": "evotac.strict_paired_prefix.v1",
              "evidence_scope": "strict paired counterfactual if replay gates pass",
              "seed": args.seed, "prefix_controls": args.prefix_controls,
              "post_prefix_controls": args.post_prefix_controls,
              "recovery_controls": args.recovery_controls,
              "monitor_object_lost_risk": args.monitor_object_lost_risk,
              "warmstart": str(args.warmstart), "trainer_checkpoint": str(args.trainer_checkpoint),
              "branches": []}
    try:
        backend = FTP1Client(policy_cfg, paths["run_root"])
        policy = FTP1Policy(backend, policy_cfg)
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        observation, _ = wrapper.reset(args.seed, split=args.split)
        policy.reset()
        for _ in range(args.prefix_controls):
            if wrapper.done:
                raise RuntimeError("source reached terminal before prefix boundary")
            observation, _, terminated, truncated, _ = wrapper.step(policy.next_action(observation, policy_cfg["prompt"]))
            if terminated or truncated:
                raise RuntimeError("source reached terminal before prefix boundary")
        scene = wrapper.export_scene()
        shared_snapshot = wrapper.snapshot_shared_state()
        pending = [action.as_dict() for action in policy.pending]
        report.update(source_episode_path=str(wrapper.logger.path), source_scene_sha256=scene["integrity_sha256"],
                      source_pending_actions=len(pending), versions=versions,
                      shared_snapshot_digest=shared_snapshot["snapshot_digest"],
                      shared_snapshot_frame=shared_snapshot["uipc_frame"])
        with h5py.File(paths["run_root"] / "source_scene.h5", "x") as handle:
            write_tree(handle, scene)

        recovery_config = yaml.safe_load((ROOT / "configs/recovery_sac_minimal.yaml").read_text())
        encoder_provenance = json.loads((ROOT / "checkpoints/univtac_release/encoder_source.json").read_text())
        encoder = CachedTactileEncoder(FrozenTactileEncoder(
            ROOT / "checkpoints/univtac_release/encoder.pth", encoder_provenance["sha256"], device="cpu"))
        learner = RecoverySAC(128, 7, hidden_dim=128, device="cpu", seed=0)
        _load_actor(args.trainer_checkpoint, args.warmstart, learner)

        # A one-step same-action probe is the minimum continuation gate. It is
        # run from two independent restores of the same UIPC frame before the
        # actual A/B branches. The probe does not prove every future step, but
        # it prevents a shared frame identifier from being mistaken for causal
        # equivalence.
        probe_action = Action(**pending[0]) if pending else None
        if probe_action is None:
            raise RuntimeError("source prefix did not leave a pending base action for continuation probe")
        probe_observations = []
        for probe_name in ("probe_baseline", "probe_recovery"):
            policy.reset()
            backend.reset_seed(args.seed)
            probe_observation = wrapper.restore_shared_state(shared_snapshot, branch_id=probe_name)
            probe_observation, _, terminated, truncated, _ = wrapper.step(probe_action)
            if probe_observation is None or terminated or truncated:
                raise RuntimeError("continuation probe terminated before producing an observation")
            probe_observations.append(probe_observation)
            wrapper.stop("continuation_probe_complete")
        continuation_probe = observation_audit(probe_observations[0], probe_observations[1])
        report["continuation_probe"] = continuation_probe

        def run_branch(name):
            policy.reset()
            backend.reset_seed(args.seed)
            observation = wrapper.restore_shared_state(shared_snapshot, branch_id=name)
            reconstruction = {"status": "shared_state_restored",
                              "snapshot_digest": shared_snapshot["snapshot_digest"],
                              "uipc_frame": shared_snapshot["uipc_frame"],
                              "continuation_probe_match": continuation_probe.get("observations_match", False),
                              **observation_audit(scene["observation"], observation)}
            row = {"branch": name, "reconstruction": reconstruction,
                   "episode_path": str(wrapper.logger.path), "recovery_actions": 0,
                   "post_prefix_controls": 0}
            if name == "baseline":
                policy.pending.extend(Action(**action) for action in pending)

            history = RecoveryHistory(length=8, physical_hz=int(config["simulation"]["physical_hz"]),
                                      task_budget_steps=int(config["budgets"]["task_physics_steps"]),
                                      recovery_budget_steps=int(config["budgets"]["recovery_physics_steps"]))
            cache = {"obs": None, "bundle": None, "state": None}

            def feature_state(obs):
                if cache["obs"] is not obs:
                    remaining = int(config["budgets"]["recovery_physics_steps"] - wrapper.recovery_elapsed)
                    cache["obs"] = obs
                    cache["bundle"] = recovery_features(obs, encoder, recovery_remaining_steps=max(0, remaining))
                    frames, mask = history.append(cache["bundle"])
                    with torch.inference_mode():
                        cache["state"] = learner_actor_encoder(frames, mask)
                return cache["state"]

            # Keep the same state encoding used by train_recovery, including a
            # fresh causal history at the fixed branch boundary.
            def learner_actor_encoder(frames, mask):
                # HistoryEncoder is loaded from the warm-start checkpoint only
                # through the trainer's actor; the actor consumes 128D states.
                # This local branch obtains the frozen encoder lazily below.
                return history_encoder(frames[None], mask[None]).squeeze(0).cpu().numpy()

            warm_state = torch.load(args.warmstart, map_location="cpu", weights_only=False)
            if warm_state.get("schema") != "evotac.control_warmstart.v1":
                raise ValueError("warmstart checkpoint schema mismatch")
            history_encoder = HistoryEncoder.from_checkpoint(warm_state["encoder"], device="cpu", frozen=True)
            history_encoder.eval()

            if name == "recovery":
                policy.switch_control()
            for control in range(args.post_prefix_controls):
                if wrapper.done:
                    break
                if name == "recovery" and control < args.recovery_controls:
                    action = Action("recovery_delta", learner.act(feature_state(observation), deterministic=True))
                    row["recovery_actions"] += 1
                else:
                    if name == "recovery" and control == args.recovery_controls:
                        policy.switch_control()
                    action = policy.next_action(observation, policy_cfg["prompt"])
                observation, _, terminated, truncated, _ = wrapper.step(action)
                row["post_prefix_controls"] = control + 1
                if terminated or truncated:
                    break
            if not wrapper.done:
                wrapper.stop("strict_paired_control_limit")
            row.update(status="finished", outcome=_outcome(wrapper))
            return row

        for name in ("baseline", "recovery"):
            report["branches"].append(run_branch(name))
        report["paired_valid"] = all(
            row.get("status") == "finished" and row.get("reconstruction", {}).get("valid_match") is True
            and report.get("continuation_probe", {}).get("observations_match") is True
            and row.get("outcome", {}).get("valid_trial") is True
            and not row.get("outcome", {}).get("incomplete", True)
            for row in report["branches"])
        report["status"] = "completed" if report["paired_valid"] else "completed_with_ineligible_branches"
        report["evidence_scope"] = "snapshot diagnostic; single-step probe does not certify hidden-state equivalence"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(report), indent=2))
        if backend is not None:
            backend.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()
    print(json.dumps({"status": report["status"], "paired_valid": report.get("paired_valid"),
                      "branches": [{"branch": x.get("branch"), "status": x.get("status"),
                                    "reason": (x.get("outcome") or {}).get("reason"),
                                    "reconstruction": (x.get("reconstruction") or {}).get("valid_match")}
                                   for x in report["branches"]]}, indent=2))


if __name__ == "__main__":
    main()
