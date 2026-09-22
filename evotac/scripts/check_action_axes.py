"""Measure signed world-frame responses at the configured action scales."""
import json
import traceback
import numpy as np

from evotac.data.schemas import Action, json_value
from evotac.envs.action_adapter import quaternion_product
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def measured_rotvec(before, after):
    inverse = np.asarray(before, dtype=float)*np.array([1, -1, -1, -1])
    delta = quaternion_product(after, inverse)
    if delta[0] < 0:
        delta = -delta
    norm = np.linalg.norm(delta[1:])
    return delta[1:] * (2 if norm < 1e-12 else 2*np.arctan2(norm, delta[0])/norm)


def main():
    args = parser_for(__doc__).parse_args()
    config, paths, versions, launcher = launch(args)
    wrapper = None
    checks = {"status": "running", "axes": []}
    try:
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        wrapper.reset(args.seed)
        for axis in range(7):
            for sign in (1., -1.):
                values = np.zeros(7)
                values[axis] = sign
                _, _, _, _, info = wrapper.step(Action("recovery_delta", values))
                execution = info["execution_info"]
                if wrapper.done:
                    raise RuntimeError(f"Axis test terminated: {execution}")
                change = execution["measured_state_change"]
                if axis < 3:
                    response = change["ee_position_world"][axis]
                elif axis < 6:
                    response = measured_rotvec(change["ee_quaternion_world_before"], change["ee_quaternion_world_after"])[axis-3]
                else:
                    response = float(np.mean(change["finger_position"]))
                record = {"axis": axis, "sign": sign, "response": float(response),
                          "physics_steps": execution["physics_steps"],
                          "sim_seconds": execution["sim_time_end"]-execution["sim_time_start"],
                          "wall_seconds": execution["wall_seconds"]}
                checks["axes"].append(record)
                assert sign*response > 0 and execution["physics_steps"] == 6, record
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
