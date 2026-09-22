"""Control-boundary recovery handoff; the wrapper retains all physics ownership."""
from evotac.data.schemas import Action
from evotac.envs.state_replay import digest
from evotac.policy.ftp1_adapter import encode_observation


def scripted_handoff(wrapper, policy, recovery_values, prompt):
    if wrapper.done or not recovery_values:
        raise ValueError("Handoff requires a live episode and recovery controls")
    before = {"physics": wrapper.task._physics_step_count, "task": wrapper.task_elapsed,
              "recovery": wrapper.recovery_elapsed, "inferences": policy.inference_count}
    cancelled = len(policy.pending)
    policy.switch_control()
    wrapper.policy_state.switch_control()
    assert not policy.pending and not wrapper.policy_state.action_queue
    assert wrapper.task._physics_step_count == before["physics"]
    wrapper.logger.event("ftp1_paused", {**before, "cancelled_policy_actions": cancelled})
    executions = []
    for values in recovery_values:
        _, _, _, _, info = wrapper.step(Action("recovery_delta", values))
        executions.append(info["execution_info"])
        if wrapper.done:
            wrapper.logger.event("handoff_terminated_during_recovery", {"reason": info["execution_info"]["reason"]})
            return None, {"status": "terminated_during_recovery", "executions": executions}
    # Both entry and exit discard queues. Keep real observation history and
    # previous accepted command; this released FTP-1 checkpoint is stateless.
    policy.switch_control()
    wrapper.policy_state.switch_control()
    observation, _ = wrapper._observe()
    wrapper.last_observation = observation
    after = int(wrapper.task._physics_step_count)
    spent = sum(row["physics_steps"] for row in executions)
    assert after-before["physics"] == spent
    assert wrapper.task_elapsed-before["task"] == spent
    assert wrapper.recovery_elapsed-before["recovery"] == spent
    assert observation["physics_step"] == after
    assert observation["remaining_physics_steps"] == wrapper.config["budgets"]["task_physics_steps"]-wrapper.task_elapsed
    assert all(m["valid"] and m["sample_physics_step"] == after
               for sensors in observation["images"].values() for fields in sensors.values() for m in fields.values())
    inputs, _ = encode_observation(observation)
    action = policy.next_action(observation, prompt)
    assert policy.inference_count == before["inferences"]+1
    assert wrapper.task._physics_step_count == after
    evidence = {"status": "fresh_action_ready", "before": before, "after_physics": after,
                "cancelled_policy_actions": cancelled, "recovery_physics_steps": spent,
                "task_elapsed": wrapper.task_elapsed, "recovery_elapsed": wrapper.recovery_elapsed,
                "remaining_physics_steps": observation["remaining_physics_steps"],
                "fresh_input_sha256": digest(inputs), "fresh_inference": policy.last_inference,
                "history_steps": [x["observation"]["physics_step"] for x in wrapper.contract.history],
                "fresh_action": action.as_dict(), "executions": executions}
    wrapper.logger.event("ftp1_resumed", evidence)
    return action, evidence
