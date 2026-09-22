"""Retain a real budget failure and a physical object-release failure via livestream."""
import json
import traceback

from evotac.data.schemas import Action, json_value
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    args = parser_for(__doc__).parse_args()
    config, paths, versions, launcher = launch(args)
    wrapper = None
    checks = {"status": "running", "outcomes": []}
    try:
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        wrapper.reset(args.seed, branch_id="budget_failure")
        while not wrapper.done:
            _, _, _, _, info = wrapper.step(Action("recovery_delta", [0]*7))
        checks["outcomes"].append({"episode": str(wrapper.logger.path), "execution_info": info["execution_info"]})
        assert info["execution_info"]["reason"] == "recovery_budget"
        assert info["execution_info"]["truncated"] and not info["execution_info"]["bootstrap_allowed"]
        wrapper.reset(args.seed, branch_id="physical_release")
        initial = wrapper.task.read_robot_state()
        while not wrapper.done:
            # A real physical gripper opening, not a rigid-body pose write.
            _, _, _, _, info = wrapper.step(Action("joint_target", [*initial.joint_position, .039]))
        checks["outcomes"].append({"episode": str(wrapper.logger.path), "execution_info": info["execution_info"]})
        assert info["execution_info"]["reason"] == "object_lost", info["execution_info"]
        checks["status"] = "passed"
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        (paths["run_root"] / "checks.json").write_text(json.dumps(json_value(checks), indent=2))
        if wrapper:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
