"""Execute a few training/dev legacy proxy targets under the new controller."""
import json
import traceback
import numpy as np

from evotac.data.schemas import Action, json_value
from evotac.data.legacy_demonstrations import LegacyDemonstrations
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--episodes", nargs="+", default=["0", "1"])
    args = parser.parse_args()
    legacy = LegacyDemonstrations()
    manifest = legacy.manifest()
    rows = [next(r for r in manifest if r["source_episode_id"] == episode) for episode in args.episodes]
    if any(r["split"] == "test" or r["source_seed"] is None for r in rows):
        raise ValueError("Use only training/dev episodes with known source seeds")
    config, paths, versions, launcher = launch(args)
    wrapper = None
    reports = []
    try:
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        for row in rows:
            report = {"source": row, "classification": "insufficient_evidence", "act_closed_loop_tested": False}
            reports.append(report)
            try:
                candidate = legacy.candidate(row)
                obs, _ = wrapper.reset(row["source_seed"], parent_scene_id=row["parent_scene_id"], split=row["split"])
                measured = np.r_[obs["proprioception"]["joint_position"], obs["proprioception"]["finger_position"].mean()]
                report["initial_joint_error"] = measured-candidate["joint_position"][0]
                tracking, clipping = [], []
                for target in candidate["future_measured_qpos"]:
                    obs, _, _, _, info = wrapper.step(Action("joint_target", target))
                    execution = info["execution_info"]
                    if obs is not None:
                        measured = np.r_[obs["proprioception"]["joint_position"], obs["proprioception"]["finger_position"].mean()]
                        tracking.append(measured-target)
                    if execution.get("commanded_action"):
                        clipping.append(execution["commanded_action"]["joint_clip_delta"])
                    if wrapper.done:
                        break
                report.update(tracking_errors=tracking, clipping=clipping, executed_targets=len(tracking),
                              final_success=bool(wrapper.task.check_success()),
                              final_execution=execution, episode_path=str(wrapper.logger.path))
                if not wrapper.done:
                    wrapper.stop("legacy_proxy_exhausted")
                # Unknown collection frequency/joint-order evidence remains a gate
                # even if the tested proxy trajectory happens to reach success.
                report["classification"] = "warm_start_candidate_only" if report["final_success"] else "incompatible_under_tested_controller"
                report["formal_training_eligible"] = row["training_eligible"]
            except Exception as exc:
                report["exception"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                break
    finally:
        (paths["run_root"] / "legacy_compatibility.json").write_text(json.dumps(json_value({
            "reports": reports, "source_rate_status": "unconfirmed", "scope": "measured future-state proxy execution, not ACT closed loop",
            "phase2": "Confirm historical timing/joint order; if tracking or coverage is insufficient collect matching 20 Hz demonstrations. Encoder/contact validation remains pending."}), indent=2))
        if wrapper:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
