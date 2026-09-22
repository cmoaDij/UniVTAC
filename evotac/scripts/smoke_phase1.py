"""Livestream integration: reset, joint/recovery controls, saving and replay."""
import json
import traceback
import time

import numpy as np

from evotac.data.schemas import Action, json_value
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--reset-only", action="store_true")
    parser.add_argument("--split", choices=("train", "dev", "test"), default=None)
    parser.add_argument("--lateral-offset", nargs=2, type=float, metavar=("X_METRES", "Y_METRES"))
    args = parser.parse_args()
    config, paths, versions, launcher = launch(args)
    wrapper = None
    checks = {"status": "running", "seed": args.seed, "backend": "taxim", "launch_mode": "livestream"}
    try:
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        condition = ({"kind": "lateral_offset", "offset_world_m": args.lateral_offset}
                     if args.lateral_offset is not None else {"kind": "nominal"})
        obs, info = wrapper.reset(args.seed, condition=condition, split=args.split)
        checks["scene"] = info["scene_metadata"]
        checks["reset"] = {"physics_step": obs["physics_step"], "ledger": wrapper.task.ledger.copy()}
        checks["rendering"] = {}
        for category, sensors in obs["images"].items():
            for sensor, fields in sensors.items():
                for field, modality in fields.items():
                    data = modality["data"]
                    record = {"shape": list(data.shape), "std": float(data.std()), "min": int(data.min()), "max": int(data.max()), "valid": modality["valid"], "sample_physics_step": modality["sample_physics_step"]}
                    checks["rendering"][f"{category}/{sensor}/{field}"] = record
                    assert modality["valid"] and data.std() > 1, record
        # Save viewable RGB evidence without introducing lossy encoding.
        import cv2
        image_dir = paths["run_root"] / "images"
        image_dir.mkdir()
        for category, sensors in obs["images"].items():
            for name, fields in sensors.items():
                for field, modality in fields.items():
                    cv2.imwrite(str(image_dir/f"{category}_{name}_{field}.png"), cv2.cvtColor(modality["data"], cv2.COLOR_RGB2BGR))
        assert set(wrapper.task.references) == {"empty", "grasp"}
        assert wrapper.task.references["empty"]["physics_step"] < wrapper.task.references["grasp"]["physics_step"] < obs["physics_step"]
        if not args.reset_only:
            steps = []
            control_wall_start = time.perf_counter()
            for _ in range(8):
                state = wrapper.task.read_robot_state()
                action = Action("joint_target", [*state.joint_position, float(state.finger_position.mean())])
                obs, _, terminated, truncated, info = wrapper.step(action)
                steps.append(info["execution_info"])
                assert not terminated and not truncated and not wrapper.done, info["execution_info"]
                assert steps[-1]["physics_steps"] == 6
            for values in ([0]*7, [.1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, .1, 0], [0, 0, 0, 0, 0, 0, .1]):
                obs, _, terminated, truncated, info = wrapper.step(Action("recovery_delta", values))
                steps.append(info["execution_info"])
                assert not wrapper.done, info["execution_info"]
            control_wall_seconds = time.perf_counter()-control_wall_start
            checks["control_throughput"] = {"wall_seconds_including_logging": control_wall_seconds,
                                             "decisions_per_wall_second": len(steps)/control_wall_seconds,
                                             "simulated_seconds": sum(s["physics_steps"] for s in steps)/120}
            checks["control_steps"] = steps
            scene = wrapper.export_scene()
            obs, report = wrapper.replay_start(scene)
            checks["reconstruction"] = report
            if report["status"] != "uncalibrated":
                assert report["valid_match"], "Replay exceeded frozen tolerances"
            # Switch only at a control boundary and prove stale queued commands
            # are cancelled without consuming physics or opening the gripper.
            boundary = wrapper.task._physics_step_count
            wrapper.policy_state.action_queue.extend([Action("recovery_delta", [1]*7)]*2)
            wrapper.policy_state.switch_control()
            assert not wrapper.policy_state.action_queue and wrapper.task._physics_step_count == boundary
            checks["control_source_switch"] = {"cancelled_commands": 2, "physics_steps": 0}
            while not wrapper.done:
                _, _, _, _, failure_info = wrapper.step(Action("recovery_delta", [0]*7))
            checks["budget_failure"] = failure_info["execution_info"]
            assert failure_info["execution_info"]["reason"] == "recovery_budget"
            assert not failure_info["execution_info"]["bootstrap_allowed"]
            checks["status"] = "controls_passed_replay_uncalibrated" if report["status"] == "uncalibrated" else "checked"
        else:
            checks["status"] = "reset_and_render_passed"
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(checks), indent=2))
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
