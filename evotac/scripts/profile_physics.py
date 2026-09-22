"""Bounded real Isaac/TacEx profile, preserving cameras and tactile outputs."""
import json
import sys
import time
import traceback
import argparse
import os

from evotac.data.schemas import Action, json_value
from evotac.config import load_config
from evotac.perf.runtime_probe import gpu_preflight, gpu_snapshot, summarize_rates
from evotac.perf.scoped_timing import ScopedTiming, instrument_wrapper
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--min-free-mib", type=int, default=20000)
    pre.add_argument("--allow-busy-gpu", action="store_true")
    pre.add_argument("--flush-every", type=int)
    pre.add_argument("--dry-run", action="store_true")
    pre_args, _ = pre.parse_known_args()
    # Refuse an obviously occupied device before importing AppLauncher.  The
    # import itself bootstraps Kit and may ask for the EULA, which is wasteful
    # when a long reset is guaranteed to fail from GPU pressure.
    if not pre_args.dry_run and not pre_args.allow_busy_gpu:
        preflight = gpu_preflight(gpu_snapshot(), device=pre_args.device, min_free_mib=pre_args.min_free_mib,
                                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        if not preflight["ok"]:
            raise RuntimeError(f"GPU preflight refused profile: {preflight}")
    parser = parser_for(__doc__, app_launcher=not pre_args.dry_run)
    parser.add_argument("--controls", type=int, default=12)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate profile settings without starting Isaac")
    parser.add_argument("--min-free-mib", type=int, default=20000,
                        help="refuse a real profile when the target GPU has less free memory")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="override the memory preflight")
    parser.add_argument("--flush-every", type=int,
                        help="performance-only HDF5 flush interval; 1 preserves durable every-step logging")
    args = parser.parse_args()
    if not 1 <= args.controls <= 20:
        raise ValueError("profile controls must be in [1, 20]")
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status": "validated", "profile": "headless" if args.performance_profile else "livestream",
                          "controls": args.controls, "physical_hz": config["simulation"]["physical_hz"],
                          "control_hz": config["simulation"]["control_hz"], "decimation": config["simulation"]["physical_hz"] // config["simulation"]["control_hz"],
                          "flush_every": args.flush_every or config["logging"]["flush_every"],
                          "cameras": config["observation"]["cameras"], "tactile": config["observation"]["tactile_sensors"]}, indent=2))
        return
    preflight = gpu_preflight(gpu_snapshot(), device=args.device, min_free_mib=args.min_free_mib,
                              visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not args.allow_busy_gpu and not preflight["ok"]:
        raise RuntimeError(f"GPU preflight refused profile: {preflight}")
    config, paths, versions, launcher = launch(args)
    wrapper = None
    timing = ScopedTiming()
    checks = {"status": "running", "seed": args.seed, "backend": "taxim",
              "launch_mode": "headless_profile" if args.performance_profile else "livestream",
              "evidence_scope": "real bounded controls; not policy success or recovery efficacy",
              "gpu_preflight": preflight, "controls": []}

    def save():
        checks["timings"] = timing.report()
        (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(checks), indent=2))

    try:
        with timing.measure("construct_task"):
            wrapper = build_wrapper(config, args.run_id, versions, args.device)
        instrument_wrapper(wrapper, timing)
        timing.phase = "reset"
        with timing.measure("reset_total"):
            observation, _ = wrapper.reset(args.seed)
        timing.wrap(wrapper.logger, "append", "logging")
        checks["episode_path"] = str(wrapper.logger.path)
        checks["reset_physics_step"] = observation["physics_step"]
        print(f"Profile reset complete: physics={observation['physics_step']}", flush=True)
        timing.phase = "controls"
        started = time.perf_counter()
        for index in range(args.controls):
            if index < 8:
                p = observation["proprioception"]
                action = Action("joint_target", [*p["joint_position"], float(p["finger_position"].mean())])
            else:
                # Same four small controls as the established phase-1 smoke.
                values = ([0]*7, [.1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, .1, 0],
                          [0, 0, 0, 0, 0, 0, .1])[(index-8) % 4]
                action = Action("recovery_delta", values)
            observation, _, _, _, info = wrapper.step(action)
            execution = info["execution_info"]
            checks["controls"].append(execution)
            save()
            print(f"Profile control {index}: {execution['status']}, {execution['physics_steps']} physics steps", flush=True)
            if execution["status"] != "executed" or not execution["valid_trial"]:
                raise RuntimeError(f"Control failed: {execution.get('reason')}")
            if wrapper.done:
                break
        checks["control_wall_seconds"] = time.perf_counter() - started
        checks["executor_rates"] = summarize_rates(checks["controls"])
        checks["end_to_end_control_hz"] = len(checks["controls"]) / checks["control_wall_seconds"]
        if args.replay and not wrapper.done:
            timing.phase = "export"
            with timing.measure("scene_export"):
                scene = wrapper.export_scene()
            timing.phase = "replay"
            with timing.measure("replay_total"):
                _, report = wrapper.replay_start(scene)
            checks["reconstruction"] = report
            checks["replay_episode_path"] = str(wrapper.logger.path)
            print(f"Profile replay v3 valid_match={report['valid_match']}", flush=True)
        wrapper.stop("performance_profile_complete")
        checks["status"] = "profile_complete"
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        checks["gpu_snapshot_after"] = gpu_snapshot()
        save()
        timing.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
