"""Run a real FTP-1 checkpoint through the EvoTac livestream control interface."""
import json
from pathlib import Path
import traceback

import yaml

from evotac.config import ROOT
from evotac.data.schemas import Action, json_value
from evotac.policy.ftp1_adapter import FTP1Policy
from evotac.policy.ftp1_client import FTP1Client
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--policy-config", type=Path, default=ROOT/"configs/ftp1_insert_hole.yaml")
    parser.add_argument("--max-controls", type=int, default=200)
    parser.add_argument("--model-gpu")
    parser.add_argument("--prompt")
    parser.add_argument("--replay-at", type=int, default=0,
                        help="Export a live prefix at this control count; replay after the baseline finishes")
    args = parser.parse_args()
    if args.max_controls < 1:
        raise ValueError("At least one policy control is required")
    policy_config = yaml.safe_load(args.policy_config.read_text())
    gripper_mode = policy_config.get("gripper_target_mode", "predicted")
    if gripper_mode not in {"predicted", "hold_initial"}:
        raise ValueError("Unknown gripper target diagnostic mode")
    policy_config["inference_seed"] = args.seed
    if args.model_gpu is not None:
        policy_config["model_gpu"] = args.model_gpu
    if args.prompt is not None:
        policy_config["prompt"] = args.prompt
    config, paths, versions, launcher = launch(args)
    backend, wrapper = None, None
    checks = {"status": "starting", "backend": "ftp1", "seed": args.seed,
              "launch_mode": "headless_profile" if args.performance_profile else "livestream",
              "controls": [], "task_success": False}
    try:
        backend = FTP1Client(policy_config, paths["run_root"])
        policy = FTP1Policy(backend, policy_config)
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        observation, _ = wrapper.reset(args.seed)
        initial_gripper = float(observation["proprioception"]["finger_position"][0])
        policy.reset()
        checks["model"] = backend.metadata
        replay_scene = None
        for index in range(args.max_controls):
            input_physics = observation["physics_step"]
            prior_count = policy.inference_count
            action = policy.next_action(observation, policy_config["prompt"])
            model_action = action.as_dict()
            if gripper_mode == "hold_initial":
                values = list(action.values)
                values[-1] = initial_gripper
                action = Action("joint_target", values)
            observation, reward, terminated, truncated, info = wrapper.step(action)
            execution = info["execution_info"]
            row = {"control": index, "execution_status": execution["status"],
                   "physics_steps": execution["physics_steps"], "reward": reward,
                   "terminated": terminated, "truncated": truncated,
                   "inference": policy.last_inference, "reason": execution.get("reason"),
                   "input_physics_step": input_physics,
                   "model_action": model_action, "gripper_target_mode": gripper_mode,
                   "new_inference": policy.inference_count != prior_count}
            checks["controls"].append(row)
            (paths["run_root"]/"checks.json").write_text(json.dumps(json_value(checks), indent=2))
            print(f"FTP-1 control {index}: {execution['status']}, physics_steps={execution['physics_steps']}", flush=True)
            if wrapper.done:
                break
            if args.replay_at and index+1 == args.replay_at:
                replay_scene = wrapper.export_scene()
        if not wrapper.done:
            wrapper.stop("policy_control_limit")
        checks.update(status="executed", task_success=bool(wrapper.task.check_success()),
                      inferences=policy.inference_count, episode_path=str(wrapper.logger.path),
                      outcome=json.loads(wrapper.logger.handle.attrs["outcome"]))
        if not checks["controls"] or not any(c["physics_steps"] == 6 for c in checks["controls"]):
            raise RuntimeError("No full policy control was executed")
        if checks["outcome"]["reason"] == "policy_control_limit":
            checks["status"] = "executed_partial"
        elif not checks["outcome"]["valid_trial"]:
            raise RuntimeError("Policy run ended in an invalid trial")
        # Baseline outcome is saved before creating a separate replay episode.
        (paths["run_root"]/"checks.json").write_text(json.dumps(json_value(checks), indent=2))
        if replay_scene is not None:
            policy.switch_control()
            _, checks["reconstruction"] = wrapper.replay_start(replay_scene)
            checks["replay_episode_path"] = str(wrapper.logger.path)
            wrapper.stop("replay_validation_complete")
        elif args.replay_at:
            checks["replay_unavailable"] = "Baseline terminated before the selected export boundary"
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if wrapper is not None and not wrapper.done:
            wrapper.logger.event("policy_exception", {"exception": checks["error"]})
            wrapper.stop("policy_exception")
        traceback.print_exc()
        raise
    finally:
        (paths["run_root"]/"checks.json").write_text(json.dumps(json_value(checks), indent=2))
        if backend is not None:
            backend.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
