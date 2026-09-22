"""Calibrate repeatability on independent dev parents; retain every replay cost."""
import json
import subprocess
import sys
import traceback

from evotac.data.schemas import Action, json_value
from evotac.envs.state_replay import calibrate_tolerances
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def calibrate_fresh_processes(args):
    """Match standalone deployment: each source parent starts a fresh Kit process."""
    from evotac.config import load_config, run_paths
    config = load_config(args.config)
    paths = run_paths(config, args.run_id)
    paths["run_root"].mkdir(parents=True, exist_ok=False)
    paths["dataset_root"].mkdir(parents=True, exist_ok=False)
    reports, children, manifest = [], [], None
    status = {"status": "running"}
    try:
        for seed in args.dev_seeds:
            child_id = f"{args.run_id}_seed{seed}"
            command = [sys.executable, "-u", "-m", "evotac.scripts.calibrate_replay",
                       "--config", str(args.config), "--run-id", child_id,
                       "--dev-seeds", str(seed), "--repeats", str(args.repeats),
                       "--tolerance-version", args.tolerance_version,
                       "--device", args.device, "--parent-worker"]
            if args.performance_profile:
                command.append("--performance-profile")
            else:
                command.extend(["--livestream", str(args.livestream)])
            with (paths["run_root"]/f"seed{seed}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            child_paths = run_paths(config, child_id)
            checks = json.loads((child_paths["run_root"]/"checks.json").read_text())
            children.append({"run_id": child_id, "returncode": result.returncode, "checks": checks})
            if result.returncode or checks["status"] != "parent_repeats_completed":
                raise RuntimeError(f"Development parent {seed} failed; see retained child artifacts")
            reports.extend(json.loads(line) for line in (child_paths["run_root"]/"reconstruction.jsonl").read_text().splitlines())
            manifest = json.loads((child_paths["dataset_root"]/"manifest.json").read_text())
            print(f"Completed fresh-process development parent {seed}", flush=True)
        parents = {r["parent_scene_id"] for r in reports}
        tolerances = calibrate_tolerances(reports, parents, args.tolerance_version)
        (paths["run_root"]/"tolerances.json").write_text(json.dumps(json_value(tolerances), indent=2))
        status = {"status": "calibrated_pending_independent_validation",
                  "development_parents": sorted(parents), "replays": len(reports),
                  "tolerance_version": args.tolerance_version, "launch_protocol": "fresh_process_per_parent"}
    except BaseException as exc:
        status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        (paths["run_root"]/"checks.json").write_text(json.dumps({**status, "children": children}, indent=2))
        (paths["run_root"]/"reconstruction.jsonl").write_text("".join(json.dumps(r)+"\n" for r in reports))
        if manifest is not None:
            manifest.update(data_kind="calibration_cohort", command=sys.argv, child_runs=children)
            (paths["dataset_root"]/"manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--dev-seeds", nargs="+", type=int, default=[0, 18, 23, 24])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tolerance-version", default="dev_repeat_max.v1")
    parser.add_argument("--fresh-processes", action="store_true")
    parser.add_argument("--parent-worker", action="store_true", help="Internal single-parent cohort worker")
    args = parser.parse_args()
    if not args.performance_profile and (args.headless or args.livestream not in (1, 2)):
        raise ValueError("Calibration requires livestream or explicit --performance-profile headless mode")
    if len(set(args.dev_seeds)) != len(args.dev_seeds):
        raise ValueError("Development seeds must be distinct")
    if len(set(args.dev_seeds)) < (1 if args.parent_worker else 2) or args.repeats < 2:
        raise ValueError("At least two distinct dev seeds and two repeats required")
    from evotac.data.legacy_demonstrations import LegacyDemonstrations
    known = {row["source_seed"]: row["split"] for row in LegacyDemonstrations().manifest()}
    if any(seed in known and known[seed] != "dev" for seed in args.dev_seeds):
        raise ValueError("Calibration seeds must inherit the development split of their source parent")
    if args.fresh_processes:
        return calibrate_fresh_processes(args)
    config, paths, versions, launcher = launch(args)
    wrapper, reports, parents = None, [], set()
    status = {"status": "running"}
    try:
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        for seed in args.dev_seeds:
            wrapper.reset(seed, split="dev")
            for _ in range(8):
                s = wrapper.task.read_robot_state()
                wrapper.step(Action("joint_target", [*s.joint_position, float(s.finger_position.mean())]))
                if wrapper.done:
                    raise RuntimeError("Development prefix ended before recovery point")
            for values in ([0]*7, [.1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, .1, 0], [0, 0, 0, 0, 0, 0, .1]):
                wrapper.step(Action("recovery_delta", values))
                if wrapper.done:
                    raise RuntimeError("Development recovery prefix ended before export")
            scene = wrapper.export_scene()
            parents.add(scene["parent_scene_id"])
            for _ in range(args.repeats):
                _, report = wrapper.replay_start(scene)
                reports.append(report)
        if not args.parent_worker:
            tolerances = calibrate_tolerances(reports, parents, args.tolerance_version)
            (paths["run_root"] / "tolerances.json").write_text(json.dumps(json_value(tolerances), indent=2))
        status = {"status": "parent_repeats_completed" if args.parent_worker else "calibrated_pending_independent_validation", "tolerance_version": args.tolerance_version,
                  "development_parents": sorted(parents), "replays": len(reports)}
    except BaseException as exc:
        status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "completed_replays": len(reports)}
        traceback.print_exc()
        raise
    finally:
        (paths["run_root"] / "checks.json").write_text(json.dumps(status, indent=2))
        if wrapper:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
