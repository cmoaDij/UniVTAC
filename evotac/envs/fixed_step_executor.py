"""Single owner of control-stage physics advancement; no planning or UIPC stepping."""
import time
import numpy as np

from evotac.envs.action_adapter import ActionRejected


def targets_match_readback(expected, accepted):
    """Compare the bits sent to the target API, including its dtype conversion."""
    return all(np.array_equal(np.asarray(accepted[key]), np.asarray(value, dtype=np.asarray(accepted[key]).dtype))
               for key, value in expected.items())


class FixedStepExecutor:
    def __init__(self, task, adapter, physical_hz=120, decimation=6):
        if type(physical_hz) is not int or physical_hz <= 0:
            raise ValueError("physical_hz must be a positive integer")
        if type(decimation) is not int or decimation <= 0:
            raise ValueError("decimation must be a positive integer")
        self.task, self.adapter = task, adapter
        self.physical_hz, self.decimation = physical_hz, decimation

    def execute(self, action, *, recorded_command=None):
        task = self.task
        start = int(task._physics_step_count)
        wall = time.perf_counter()
        before = task.read_robot_state()
        result = {"proposed_action": action.as_dict(), "commanded_action": None,
                  "applied_action": {"accepted": False}, "physics_start": start,
                  "status": "rejected", "reason": None, "valid_trial": True}
        try:
            command = self.adapter.adapt(action, before) if recorded_command is None else recorded_command
            result["commanded_action"] = command.as_dict()
            if not task.plan_success:
                raise ActionRejected("plan_success_false")
            # The task confirms target buffers after the setters return. This is
            # acceptance by the target API, never evidence of measured motion.
            result["applied_action"] = task.apply_command(command)
            if result["applied_action"]["accepted"] is not True:
                raise ActionRejected("target_acceptance_unconfirmed")
            with task._configured_decimation():
                task._step(is_save=False)
            elapsed = int(task._physics_step_count) - start
            if elapsed != self.decimation:
                raise ActionRejected(f"physics_step_mismatch:{elapsed}")
            if task._last_render_physics_step != task._physics_step_count:
                task._update_render()
            after = task.read_robot_state()
            result["measured_state_change"] = {
                "joint_position": after.joint_position - before.joint_position,
                "finger_position": after.finger_position - before.finger_position,
                "ee_position_world": after.ee_position_world - before.ee_position_world,
                "ee_quaternion_world_before": before.ee_quaternion_world,
                "ee_quaternion_world_after": after.ee_quaternion_world,
            }
            result["status"] = "executed"
        except ActionRejected as exc:
            result["reason"] = str(exc)
        except Exception as exc:
            result.update(status="infrastructure_error", reason=f"{type(exc).__name__}: {exc}", valid_trial=False)
        finally:
            end = int(task._physics_step_count)
            applied = result["applied_action"]
            if applied.get("accepted") is True:
                applied["accepted_by_target_api"] = True
                # A completed physics step is downstream of scene.write_data_to_sim.
                # With no step, retain API readback but do not claim confirmed
                # submission to the simulated controller after an exception.
                applied["simulation_submission_confirmed"] = end > start
                if end == start:
                    applied["accepted"] = None
            result.update(execution_reason=result["reason"], physics_end=end, physics_steps=end-start,
                          sim_time_start=start/self.physical_hz, sim_time_end=end/self.physical_hz,
                          wall_seconds=time.perf_counter()-wall)
            if result["wall_seconds"] > 0 and result["physics_steps"] > 0:
                result["wall_control_hz"] = 1.0 / result["wall_seconds"]
                result["wall_physics_hz"] = result["physics_steps"] / result["wall_seconds"]
            else:
                result["wall_control_hz"] = None
                result["wall_physics_hz"] = None
        return result
