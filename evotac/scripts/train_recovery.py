"""Run real FTP-1 plus one EvoTac recovery skill through the Isaac wrapper."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path

import torch
import yaml

from evotac.config import ROOT
from evotac.data.schemas import json_value
from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_buffer import ReplayBuffer
from evotac.learning.recovery_observation import RECOVERY_OBSERVATION_DIM, RecoveryHistory, recovery_summary
from evotac.learning.recovery_rollout import RecoveryCollector
from evotac.learning.recovery_sac import RecoverySAC
from evotac.learning.recovery_trainer import RecoverySACTrainer
from evotac.learning.recovery_training import RecoveryTrainingDriver
from evotac.learning.recovery_experiment import (
    scene_splits, validate_parents, load_replay, checkpoint_contract,
    load_training_checkpoint, save_training_checkpoint, evaluate_episode, sha256,
)
from evotac.learning.recovery_warmstart import RecoveryWarmStart
from evotac.learning.runtime_monitor import RecoveryMonitor
from evotac.learning.trigger_calibration import load_calibration
from evotac.evaluation.snapshot_audit import load_test_gate, frozen_protocol
from evotac.perf.runtime_probe import gpu_preflight, gpu_snapshot
from evotac.policy.ftp1_adapter import FTP1Policy
from evotac.policy.ftp1_client import FTP1Client
from evotac.policy.tactile_features import CachedTactileEncoder, FrozenTactileEncoder, recovery_features
from evotac.scripts.runtime import build_wrapper, launch, parser_for


def initialize_models(recovery_config, *, checkpoint=None, allow_uninitialized=False,
                      seed=0, skill_name=None):
    """Load the warm-start writer's actual output, including the selected actor."""
    device = recovery_config.get("device", "cpu")
    learner = RecoverySAC(128, int(recovery_config["action_dim"]),
                          hidden_dim=int(recovery_config["hidden_dim"]), device=device, seed=seed)
    if checkpoint is None:
        if not allow_uninitialized:
            raise ValueError("a trained warm-start checkpoint is required; --allow-uninitialized-history is plumbing only")
        encoder = HistoryEncoder(input_dim=RECOVERY_OBSERVATION_DIM, output_dim=128,
                                 hidden_dim=128).to(device).freeze()
        return encoder, learner, {"status": "uninitialized_plumbing_only",
                                  "skill_name": skill_name or "single_recovery_skill"}
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("schema") == "evotac.control_warmstart.v1":
        model = RecoveryWarmStart.from_checkpoint(state, device=device)
        if skill_name is None:
            if len(model.skill_names) != 1:
                raise ValueError("--skill-name is required for a multi-skill warm-start")
            skill_name = model.skill_names[0]
        if skill_name not in model.skill_names:
            raise ValueError("selected skill is absent from the warm-start checkpoint")
        learner.actor.load_state_dict(model.actors[skill_name].state_dict(), strict=True)
        encoder = model.encoder
        status = "frozen_warmstart_checkpoint"
    elif state.get("schema") == "evotac.history_encoder.v1" and allow_uninitialized:
        # A bare state dict has no training-parent provenance. Keep it usable
        # for explicit plumbing runs without calling it a trained model.
        encoder = HistoryEncoder.from_checkpoint(state, device=device, frozen=True)
        status = "unverified_history_plumbing_only"
        skill_name = skill_name or "single_recovery_skill"
    else:
        raise ValueError("expected a trained control warm-start with train-parent provenance")
    if (encoder.config["input_dim"] != RECOVERY_OBSERVATION_DIM
            or encoder.config["output_dim"] != 128 or encoder.config["history_length"] != 8):
        raise ValueError("history checkpoint dimensions do not match the recovery contract")
    return encoder, learner, {"status": status, "skill_name": skill_name,
                              "checkpoint": str(checkpoint),
                              "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                              "provenance": state.get("provenance", [])}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--min-free-mib", type=int, default=20000)
    pre.add_argument("--allow-busy-gpu", action="store_true")
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--history-checkpoint", type=Path)
    pre.add_argument("--allow-uninitialized-history", action="store_true")
    pre_args, _ = pre.parse_known_args()
    if not pre_args.dry_run and not pre_args.allow_busy_gpu:
        guard = gpu_preflight(gpu_snapshot(), device=pre_args.device,
                              min_free_mib=pre_args.min_free_mib,
                              visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        if not guard["ok"]:
            raise RuntimeError(f"GPU preflight refused recovery training: {guard}")
    parser = parser_for(__doc__, app_launcher=not pre_args.dry_run)
    # AppLauncher owns --device in a real Isaac invocation.  The dry-run
    # parser skips AppLauncher, so it needs the option locally instead.
    if pre_args.dry_run:
        parser.add_argument("--device", default=pre_args.device)
    parser.add_argument("--policy-config", type=Path, default=ROOT / "configs/ftp1_insert_hole_chunk16.yaml")
    parser.add_argument("--recovery-config", type=Path, default=ROOT / "configs/recovery_sac_minimal.yaml")
    parser.add_argument("--encoder-checkpoint", type=Path, default=ROOT / "checkpoints/univtac_release/encoder.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--model-gpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=20000)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--history-checkpoint", type=Path, default=pre_args.history_checkpoint)
    parser.add_argument("--trainer-checkpoint", type=Path,
                        help="resume SAC learner and replay buffer from a prior training run")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--baseline", action="store_true", help="disable recovery in the same evaluation loop")
    parser.add_argument("--seeds", type=int, nargs="+", help="explicit parent seeds, checked before Isaac startup")
    parser.add_argument("--learner-seed", type=int, default=0)
    parser.add_argument("--replay-dataset", type=Path, help="seed training replay using the same frozen encoder")
    parser.add_argument("--learning-starts", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skill-name", help="select an actor from a multi-skill warm-start")
    parser.add_argument("--deterministic-recovery", action="store_true",
                        help="use the actor mean during evaluation instead of stochastic sampling")
    parser.add_argument("--split", choices=("train", "dev", "test"), default=None,
                        help="persisted parent split for this online collection episode")
    parser.add_argument("--allow-uninitialized-history", action="store_true",
                        help="plumbing smoke only; not an effectiveness run")
    parser.add_argument("--monitor-object-lost-risk", type=float, default=0.8,
                        help="observable trigger threshold; default proxy is uncalibrated")
    parser.add_argument("--monitor-contact-blocked", type=float, default=0.8,
                        help="observable contact trigger threshold; default proxy is uncalibrated")
    parser.add_argument("--trigger-calibration", type=Path,
                        help="frozen train-split trigger calibration artifact")
    parser.add_argument("--causal-audit", type=Path,
                        help="completed strict dev/acceptance audit required before test")
    parser.add_argument("--stable-cycles", type=int, default=3,
                        help="consecutive stable observations required before handoff")
    parser.add_argument("--max-recovery-actions", type=int, default=None,
                        help="override recovery action cap; default is recovery_physics_steps // 6")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="temporary learner batch override; use 1 only for plumbing smoke")
    parser.add_argument("--updates-per-interaction", type=int, default=None,
                        help="temporary learner update-ratio override")
    args = parser.parse_args()
    if args.baseline and args.mode != "evaluate":
        raise ValueError("--baseline requires --mode evaluate")
    if args.mode == "evaluate" and args.replay_dataset is not None:
        raise ValueError("evaluation cannot load a training replay dataset")
    if args.replay_dataset is not None and args.trainer_checkpoint is not None:
        raise ValueError("resume from trainer OR seed replay, not both")
    if args.learning_starts < 1 or args.threads < 1:
        raise ValueError("learning-starts and threads must be positive")
    torch.set_num_threads(args.threads)
    if type(args.episodes) is not int or args.episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if type(args.start_seed) is not int or args.start_seed < 0:
        raise ValueError("start-seed must be a nonnegative integer")
    seeds = args.seeds or list(range(args.start_seed, args.start_seed + args.episodes))
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("episode seeds must be in [0, 2**32)")
    assigned_splits = validate_parents(seeds, scene_splits(args.config), args.mode, args.split)
    if type(args.stable_cycles) is not int or args.stable_cycles < 1:
        raise ValueError("stable-cycles must be a positive integer")
    if args.max_recovery_actions is not None and (type(args.max_recovery_actions) is not int
                                                   or args.max_recovery_actions < 1):
        raise ValueError("max-recovery-actions must be a positive integer")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in
               (args.monitor_object_lost_risk, args.monitor_contact_blocked)):
        raise ValueError("monitor thresholds must be finite and in [0, 1]")
    calibration = load_calibration(args.trigger_calibration, expected_split=args.split) if args.trigger_calibration else None
    if args.mode == "evaluate" and set(assigned_splits) & {"dev", "test"} and calibration is None:
        raise ValueError("dev/test evaluation requires --trigger-calibration fitted on train")
    if args.batch_size is not None and (type(args.batch_size) is not int or args.batch_size < 1):
        raise ValueError("batch-size must be a positive integer")
    if args.updates_per_interaction is not None and (type(args.updates_per_interaction) is not int or args.updates_per_interaction < 1):
        raise ValueError("updates-per-interaction must be a positive integer")
    policy_config = yaml.safe_load(args.policy_config.read_text())
    recovery_config = yaml.safe_load(args.recovery_config.read_text())
    controls = {
        "monitor_object_lost_risk": calibration.threshold if calibration else args.monitor_object_lost_risk,
        "monitor_contact_blocked": calibration.contact_threshold if calibration else args.monitor_contact_blocked,
        "requested_monitor_object_lost_risk": args.monitor_object_lost_risk,
        "requested_monitor_contact_blocked": args.monitor_contact_blocked,
        "trigger_calibration": str(args.trigger_calibration.resolve()) if args.trigger_calibration else None,
        "trigger_calibration_sha256": sha256(args.trigger_calibration) if args.trigger_calibration else None,
        "stable_cycles": args.stable_cycles,
        "max_recovery_actions": int(args.max_recovery_actions or 0),
        "batch_size": int(args.batch_size or recovery_config["batch_size"]),
        "updates_per_interaction": int(args.updates_per_interaction or recovery_config["updates_per_interaction"]),
        "split": args.split,
        "mode": args.mode, "baseline": args.baseline,
        "learning_starts": args.learning_starts, "learner_seed": args.learner_seed,
            # Dev/test protocols intentionally freeze the action cap and
            # stability window.  Only training-side temporary overrides are
            # plumbing evidence; evaluation records remain comparable.
            "plumbing_trigger_override": bool(args.mode != "evaluate" and (
                                         args.monitor_object_lost_risk != 0.8
                                         or args.monitor_contact_blocked != 0.8
                                         or args.stable_cycles != 3 or args.max_recovery_actions is not None
                                         or args.batch_size is not None
                                         or args.updates_per_interaction is not None)),
    }
    # Fingerprint the effective protocol. Seeds are disjoint statistical units,
    # while the code, policy and controls must remain frozen across dev/test.
    protocol_files = {"simulation_config": args.config, "policy_config": args.policy_config,
                      "recovery_config": args.recovery_config}
    protocol_files.update({str(p.relative_to(ROOT)): p for p in ROOT.rglob("*.py")
                           if p.relative_to(ROOT).parts[0] in
                           {"scripts", "learning", "policy", "envs", "evaluation", "data", "perf"}})
    protocol = frozen_protocol(protocol_files, {
        key: controls[key] for key in ("monitor_object_lost_risk", "monitor_contact_blocked",
                                      "trigger_calibration_sha256", "stable_cycles", "max_recovery_actions")})
    causal_gate = None
    if args.mode == "evaluate" and "test" in assigned_splits:
        if args.causal_audit is None:
            raise ValueError("test evaluation requires --causal-audit from a strict dev/acceptance audit")
        if args.trainer_checkpoint is None or args.history_checkpoint is None:
            raise ValueError("test evaluation requires frozen trainer and warm-start checkpoints")
        causal_gate = load_test_gate(args.causal_audit,
            warmstart_sha256=sha256(args.history_checkpoint),
            checkpoint_sha256=sha256(args.trainer_checkpoint),
            expected_protocol=protocol, test_seeds=seeds)
    controls["causal_audit_gate"] = causal_gate
    controls["frozen_protocol"] = protocol
    if int(recovery_config["observation_dim"]) != 128:
        raise ValueError("recovery configuration must use the frozen 128D history state")
    if args.dry_run:
        print(json.dumps({"status": "validated", "episodes": len(seeds), "seeds": seeds,
                          "assigned_splits": assigned_splits,
                          "observation_dim": recovery_config["observation_dim"],
                          "action_dim": recovery_config["action_dim"],
                          "training_controls": controls, "history_checkpoint_required": True,
                          "real_interaction_required": recovery_config["real_interaction_required"]}, indent=2))
        return
    guard = gpu_preflight(gpu_snapshot(), device=args.device, min_free_mib=args.min_free_mib,
                          visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    history_encoder, learner, initialization = initialize_models(
        recovery_config, checkpoint=args.history_checkpoint,
        allow_uninitialized=args.allow_uninitialized_history, seed=args.learner_seed,
        skill_name=args.skill_name)
    config, paths, versions, launcher = launch(args)
    backend = wrapper = None
    checks = {"status": "starting", "backend": "ftp1_plus_recovery_sac", "episodes": [],
              "gpu_preflight": guard, "recovery_config": recovery_config,
              "training_controls": controls, "initialization": initialization,
              "ftp1_inference_seed_rule": "episode_seed_v1",
              "deterministic_recovery": bool(args.mode == "evaluate" or args.deterministic_recovery),
              "history_encoder_status": initialization["status"], "events": [],
              "evidence_scope": "plumbing_only" if controls["plumbing_trigger_override"]
              or initialization["status"] != "frozen_warmstart_checkpoint" else "online_training_development"}
    try:
        # Match the standalone FTP-1 baseline's RNG protocol.  The worker
        # starts with this seed, and each later episode is reset explicitly
        # below before its scene is created.
        policy_config["inference_seed"] = seeds[0]
        if args.model_gpu is not None:
            policy_config["model_gpu"] = args.model_gpu
        backend = FTP1Client(policy_config, paths["run_root"])
        encoder_provenance = json.loads((ROOT / "checkpoints/univtac_release/encoder_source.json").read_text())
        encoder = CachedTactileEncoder(FrozenTactileEncoder(
            args.encoder_checkpoint, encoder_provenance["sha256"], device="cpu"))
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        buffer = ReplayBuffer(int(recovery_config["replay_capacity"]), seed=args.learner_seed)
        torch.save(history_encoder.checkpoint(), paths["run_root"] / "history_encoder.pt")
        history = RecoveryHistory(length=8, physical_hz=int(config["simulation"]["physical_hz"]),
                                  task_budget_steps=int(config["budgets"]["task_physics_steps"]),
                                  recovery_budget_steps=int(config["budgets"]["recovery_physics_steps"]))

        feature_cache = {"observation": None, "features": None, "state": None}

        def feature_bundle(observation):
            # The driver asks for a summary, a policy action, and a collector
            # state for the same observation.  Encoding it once preserves the
            # observable contract while avoiding three six-image forwards.
            if feature_cache["observation"] is observation:
                return feature_cache["features"]
            remaining = int(config["budgets"]["recovery_physics_steps"] - wrapper.recovery_elapsed)
            value = recovery_features(observation, encoder, recovery_remaining_steps=max(0, remaining))
            feature_cache["observation"], feature_cache["features"], feature_cache["state"] = observation, value, None
            return value

        def feature_state(observation):
            feature_bundle(observation)
            if feature_cache["observation"] is observation and feature_cache["state"] is not None:
                return feature_cache["state"]
            frames, mask = history.append(feature_cache["features"])
            with torch.inference_mode():
                state = history_encoder(frames[None], mask[None]).squeeze(0).cpu().numpy()
            feature_cache["observation"], feature_cache["state"] = observation, state
            return state

        def feature_summary(observation):
            bundle = feature_bundle(observation)
            feature_state(observation)
            summary = recovery_summary(bundle)
            if args.baseline:
                summary.update(object_lost_risk=-1.0, contact_blocked=-1.0)
            return summary

        def recovery_policy(observation, summary, skill):
            return learner.act(feature_state(observation),
                               deterministic=args.mode == "evaluate" or args.deterministic_recovery)

        base_policy_impl = FTP1Policy(backend, policy_config)

        def base_policy(observation):
            return base_policy_impl.next_action(observation, policy_config["prompt"])

        def on_event(kind, details):
            checks["events"].append({"kind": kind, "details": json_value(details)})
            wrapper.logger.event(kind, details)
            if kind == "recovery_trigger":
                base_policy_impl.switch_control()

        def on_handoff(observation):
            base_policy_impl.switch_control()

        monitor = RecoveryMonitor(object_lost_risk=args.monitor_object_lost_risk,
                                  contact_blocked=args.monitor_contact_blocked,
                                  calibration=calibration)
        max_recovery_actions = int(args.max_recovery_actions or
                                   (config["budgets"]["recovery_physics_steps"] // 6))
        if args.max_recovery_actions is not None and max_recovery_actions * 6 >= config["budgets"]["recovery_physics_steps"]:
            raise ValueError("explicit action cap must leave budget for handoff; use at most 19")
        driver = RecoveryTrainingDriver(
            wrapper, RecoveryCollector(wrapper, buffer,
                                       feature_state),
            feature_summary=feature_summary, base_policy=base_policy,
            recovery_policy=recovery_policy,
            max_recovery_actions=max_recovery_actions,
            handoff_at_cap=args.max_recovery_actions is not None,
            monitor=monitor, stable_cycles=args.stable_cycles,
            skill_name=initialization["skill_name"],
            handoff=on_handoff, on_event=on_event)
        trainer = RecoverySACTrainer(driver, learner, batch_size=controls["batch_size"],
                                     updates_per_interaction=controls["updates_per_interaction"],
                                     learning_starts=args.learning_starts)
        contract = checkpoint_contract(args.history_checkpoint, initialization["skill_name"], controls) if args.history_checkpoint else {}
        evidence = {"split": "train", "episodes": [], "updates": 0, "new_transitions": 0,
                    "learner_seed": args.learner_seed}
        if args.replay_dataset is not None:
            evidence["initial_replay"] = load_replay(args.replay_dataset, buffer, args.history_checkpoint)
        if args.trainer_checkpoint is not None:
            evidence = load_training_checkpoint(args.trainer_checkpoint, trainer, contract,
                                                evaluation=args.mode == "evaluate")
            checks["trainer_initialization"] = {
                "status": "resumed_trainer_checkpoint",
                "checkpoint": str(args.trainer_checkpoint),
                "sha256": sha256(args.trainer_checkpoint),
            }
        checks["evidence_scope"] = "frozen_evaluation" if args.mode == "evaluate" else "online_training_development"
        checks["trigger_provenance"] = monitor.provenance()
        checks["training_evidence"] = evidence
        for offset, episode_seed in enumerate(seeds):
            # A reset starts a fresh control episode; no pending FTP-1 chunk
            # may cross the episode boundary.
            backend.reset_seed(episode_seed)
            checks["events"].append({"kind": "ftp1_seed_reset",
                                     "details": {"seed": episode_seed}})
            base_policy_impl.reset()
            history.reset()
            feature_cache.update(observation=None, features=None, state=None)
            try:
                if args.mode == "evaluate":
                    episode = evaluate_episode(driver, learner, episode_seed, split=assigned_splits[offset])
                    buffer._items.clear()  # evaluation fragments never enter a saved training buffer
                else:
                    episode = trainer.run_episode(episode_seed, split=assigned_splits[offset]).as_dict()
            except Exception as exc:
                # Only these known physical reset failures are recoverable.
                # Learner, sensor, CUDA and programming errors must stop the run.
                known_reset_errors = {
                    "Physical grasp was lost before the grasp reference",
                    "Physical grasp was lost during approach",
                }
                logged_outcome = (json.loads(wrapper.logger.handle.attrs.get("outcome", "{}"))
                                  if wrapper.logger is not None else {})
                if (not isinstance(exc, RuntimeError) or str(exc) not in known_reset_errors
                        or logged_outcome.get("reason") != "reset_exception" or not wrapper.done):
                    raise
                driver.collector.discard()
                if args.mode == "evaluate":
                    buffer._items.clear()
                # A single Isaac/UIPC reset can fail before a valid trial
                # exists. Preserve the failure as an auditable episode and
                # continue the independent seed cohort; never turn it into a
                # replay transition or a task label.
                episode = {
                    "status": "invalid_environment_episode",
                    "reason": "reset_exception",
                    "recovery_actions": 0,
                    "admitted_transitions": 0,
                    "terminal_outcome": {"reason": "reset_exception", "valid_trial": False,
                                          "terminated": False, "truncated": False,
                                          "incomplete": True, "exception": f"{type(exc).__name__}: {exc}"},
                    "updates": [], "skipped_updates": 0,
                }
                checks["events"].append({"kind": "invalid_environment_episode",
                                         "details": {"seed": episode_seed,
                                                     "exception": f"{type(exc).__name__}: {exc}"}})
            episode.update(seed=episode_seed, split=assigned_splits[offset],
                           episode_path=str(wrapper.logger.path), ftp1_inference_seed=episode_seed)
            checks["episodes"].append(episode)
            if args.mode == "train":
                evidence["episodes"].append({"seed": episode_seed, "split": assigned_splits[offset],
                                             "episode_path": str(wrapper.logger.path)})
                evidence["updates"] += len(episode["updates"])
                evidence["new_transitions"] += episode["admitted_transitions"]
                save_training_checkpoint(paths["run_root"] / "recovery_trainer.pt", trainer, contract, evidence)
            (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(checks), indent=2))
        checks["status"] = "completed"
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        checks["gpu_snapshot_after"] = gpu_snapshot()
        (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(checks), indent=2))
        if backend is not None:
            backend.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
