"""Replay a saved FTP-1 chunk16 prefix without starting the policy worker.

Chunk16 baseline episodes already contain the accepted command stream.  This
entry point reconstructs the exported scene from that durable HDF5 record and
runs only the physical prefix replay, so diagnostics do not consume model
GPU memory or rerun the policy.  It is intentionally separate from the
baseline runner and never changes frozen replay tolerances.
"""
import argparse
import json
import traceback
from pathlib import Path

import h5py

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree, write_tree
from evotac.data.schemas import freeze, json_value
from evotac.envs.state_replay import scene_digest, seal_scene, validate_scene
from evotac.scripts.runtime import build_wrapper, launch, parser_for


def _episode_path(source_run_id):
    checks_path = output_path(Path("runs/phase1") / source_run_id / "checks.json")
    checks = json.loads(checks_path.read_text())
    if checks.get("outcome", {}).get("reason") not in {"success", "object_lost", "task_budget", "policy_control_limit"}:
        raise ValueError(f"Source run is not a valid completed rollout: {source_run_id}")
    return Path(checks["episode_path"]), checks


def _physical_device(manifest):
    """Resolve the physical GPU behind a logical CUDA device in a manifest."""
    device = str(manifest.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        return device
    try:
        logical = int(device.split(":", 1)[1]) if ":" in device else 0
    except ValueError:
        return device
    visible = manifest.get("launch_environment", {}).get("CUDA_VISIBLE_DEVICES")
    if visible:
        mapping = [item.strip() for item in str(visible).split(",") if item.strip()]
        if 0 <= logical < len(mapping):
            return f"cuda:{mapping[logical]}"
    return f"cuda:{logical}"


def hardware_comparison(source_run_id, current_manifest):
    """Record physical GPU and launch-mode compatibility for replay evidence."""
    source_path = ROOT / "datasets/phase1" / source_run_id / "manifest.json"
    try:
        source = json.loads(source_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"available": False, "reason": "source_manifest_unavailable"}
    source_device, current_device = _physical_device(source), _physical_device(current_manifest)
    return {
        "available": True,
        "source_physical_device": source_device,
        "current_physical_device": current_device,
        "same_physical_device": source_device == current_device,
        "source_launch_mode": source.get("launch_mode"),
        "current_launch_mode": current_manifest.get("launch_mode"),
        "same_launch_mode": source.get("launch_mode") == current_manifest.get("launch_mode"),
        "source_cuda_visible_devices": source.get("launch_environment", {}).get("CUDA_VISIBLE_DEVICES"),
        "current_cuda_visible_devices": current_manifest.get("launch_environment", {}).get("CUDA_VISIBLE_DEVICES"),
    }


def build_scene(source_run_id, prefix_controls=12):
    episode, checks = _episode_path(source_run_id)
    with h5py.File(episode, "r") as handle:
        reset_event = None
        for key in handle["events"]:
            event = read_tree(handle["events"][key])
            if event.get("kind") == "reset_complete":
                reset_event = event
                break
        if reset_event is None:
            raise ValueError("Source episode has no reset_complete event")
        n = len(handle["actions"])
        if n < prefix_controls:
            raise ValueError(f"Source episode has {n} actions, need {prefix_controls}")
        observations = read_tree(handle["observations"][str(prefix_controls)])
        training = read_tree(handle["training_info"][str(prefix_controls - 1)])
        controls = []
        for index in range(prefix_controls):
            record = read_tree(handle["actions"][str(index)])
            if record.get("status") != "executed":
                raise ValueError(f"Source control {index} was not executed")
            controls.append({"action": freeze(record["proposed_action"]), "execution": freeze(record)})
        metadata = json.loads(handle.attrs["scene_metadata"])
        history = []
        for index in range(max(0, prefix_controls - 8), prefix_controls):
            history.append({"observation": freeze(read_tree(handle["observations"][str(index)])),
                            "action": freeze(read_tree(handle["actions"][str(index)])["proposed_action"])})
        scene = {
            **metadata,
            "prefix": freeze(reset_event["prefix"]),
            "initial_prefix_state": freeze(reset_event["initial_prefix_state"]),
            "initial_prefix_physics": 27,
            "task_start_physics": int(read_tree(handle["observations"]["0"])["physics_step"]),
            "observation": freeze(observations),
            "references": freeze(read_tree(handle["references"])),
            "training_info": freeze(training),
            "history": freeze(history),
            "policy_state": {"history": freeze(history), "history_length": 8},
            "control_prefix": controls,
            "task_elapsed_steps": int(prefix_controls * 6),
            "recovery_elapsed_steps": 0,
            "remaining_task_steps": 1200 - int(prefix_controls * 6),
            "rng_protocol": "numpy_default_rng_seed_before_reset.v1",
            "stable_physics_steps": 32,
        }
    return seal_scene(scene), episode, checks


def _episode_tree_match(reference_path, candidate_path):
    """Compare two independently recorded current-code replay episodes.

    A replay report compares an execution with the historical source.  For a
    same-code repeat we need a separate comparison that does not involve the
    old source values.  Hashing each durable tree gives an exact, compact
    determinism check without expanding image arrays into JSON.
    """
    groups = ("observations", "actions", "training_info", "events")

    def canonical(value, group):
        if isinstance(value, dict):
            ignored = {"wall_seconds", "wall_control_hz", "wall_physics_hz"}
            if group == "events":
                ignored |= {"ledger"}
            return {key: canonical(child, group) for key, child in value.items() if key not in ignored}
        if isinstance(value, list):
            return [canonical(child, group) for child in value]
        return value
    result = {"available": True, "exact_match": True, "groups": {}}
    try:
        with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
            for group in groups:
                if group not in reference or group not in candidate:
                    result["available"] = False
                    result["exact_match"] = False
                    result["groups"][group] = {"available": False}
                    continue
                left_keys, right_keys = sorted(reference[group]), sorted(candidate[group])
                key_match = left_keys == right_keys
                mismatches = []
                for key in sorted(set(left_keys) & set(right_keys)):
                    left = canonical(read_tree(reference[group][key]), group)
                    right = canonical(read_tree(candidate[group][key]), group)
                    if scene_digest(left) != scene_digest(right):
                        mismatches.append(str(key))
                exact = key_match and not mismatches
                result["groups"][group] = {
                    "exact_match": exact,
                    "reference_count": len(left_keys),
                    "candidate_count": len(right_keys),
                    "mismatched_keys": mismatches,
                }
                result["exact_match"] &= exact
    except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
        result.update(available=False, exact_match=False, error=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--prefix-controls", type=int, default=12)
    parser.add_argument("--diagnostic-code-drift", action="store_true",
                        help="Measure old-code prefixes without claiming version-compatible replay")
    parser.add_argument("--diagnostic-implementation-drift", action="store_true",
                        help="Measure prefixes after explicit EvoTac/TacEx implementation changes; never certify replay")
    parser.add_argument("--repeat-current", type=int, default=0,
                        help="Repeat the immutable source prefix on the same GPU/code")
    parser.add_argument("--trace-reset", action="store_true",
                        help="Save offline UIPC positions/velocities around reset to isolate divergence")
    args = parser.parse_args()
    if not 0 <= args.repeat_current <= 10:
        raise ValueError("repeat-current must be in [0, 10]")
    config, paths, versions, launcher = launch(args)
    wrapper = None
    reset_trace = None
    checks = {"status": "starting", "backend": "replay_only", "source_run_id": args.source_run_id,
              "seed": args.seed, "prefix_controls": args.prefix_controls,
              "evidence_scope": "replay diagnostics only; not recovery efficacy",
              "current_repeats": []}

    def save():
        (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(checks), indent=2))

    try:
        scene, episode, source_checks = build_scene(args.source_run_id, args.prefix_controls)
        current_manifest = json.loads((paths["dataset_root"] / "manifest.json").read_text())
        checks["hardware_comparison"] = hardware_comparison(args.source_run_id, current_manifest)
        checks["version_comparison"] = {
            "diagnostic_code_drift": args.diagnostic_code_drift,
            "diagnostic_implementation_drift": args.diagnostic_implementation_drift,
            "source_versions": scene["versions"], "execution_versions": versions,
        }
        save()
        validate_scene(scene, versions, {}, diagnostic_code_drift=args.diagnostic_code_drift,
                       diagnostic_implementation_drift=args.diagnostic_implementation_drift)
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        if args.trace_reset:
            from evotac.perf.reset_trace import ResetTrace
            trace_path = paths["run_root"] / "reset_trace.h5"
            reset_trace = ResetTrace(wrapper.task, trace_path)
            checks["reset_trace_path"] = str(trace_path)
        _, report = wrapper.replay_start(
            scene,
            diagnostic_code_drift=args.diagnostic_code_drift,
            diagnostic_implementation_drift=args.diagnostic_implementation_drift,
        )
        checks.update(status="replayed", source_episode_path=str(episode), reconstruction=report,
                      replay_episode_path=str(wrapper.logger.path), outcome="replay_prefix_complete")
        save()
        print(f"Source prefix: observable_match={report['observable_match']}, "
              f"versions_match={report['versions_match']}", flush=True)
        if args.repeat_current:
            # Reuse the same immutable command stream for every repeat.
            # The task ledger is cumulative by design; differences in that
            # accounting field alone do not imply different execution.
            current_reference_episode_path = str(wrapper.logger.path)
            checks["current_reference_episode_path"] = current_reference_episode_path
            for index in range(args.repeat_current):
                _, repeated = wrapper.replay_start(
                    scene,
                    diagnostic_code_drift=args.diagnostic_code_drift,
                    diagnostic_implementation_drift=args.diagnostic_implementation_drift,
                )
                repeat_path = str(wrapper.logger.path)
                repeat_record = {
                    "index": index, "reconstruction": repeated,
                    "replay_episode_path": repeat_path,
                    "same_code_execution": repeated.get("execution_versions") == versions,
                    "same_code_as_reference": _episode_tree_match(current_reference_episode_path, repeat_path),
                }
                checks["current_repeats"].append(repeat_record)
                save()
                print(f"Same GPU/code repeat {index}: valid_match={repeated['valid_match']}, "
                      f"exact_match={repeat_record['same_code_as_reference']['exact_match']}", flush=True)
        wrapper.stop("replay_validation_complete")
        with h5py.File(paths["run_root"] / "scene.h5", "x") as handle:
            write_tree(handle, scene)
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if wrapper is not None and not wrapper.done:
            wrapper.stop("replay_exception")
        traceback.print_exc()
        raise
    finally:
        save()
        if reset_trace is not None:
            reset_trace.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()
    print(json.dumps({"run_id": args.run_id, "source_run_id": args.source_run_id,
                      "valid_match": checks["reconstruction"]["valid_match"]}, indent=2))


if __name__ == "__main__":
    main()
