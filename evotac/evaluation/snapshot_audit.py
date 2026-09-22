"""Measure snapshot restoration; a shared identifier is never match evidence."""
from evotac.envs.state_replay import array_error, scene_digest


def observation_audit(source, candidate):
    """Conservative exact-observation probe, with errors for diagnosing failures.

    No camera tolerance has been calibrated for snapshot restore. Exact image
    equality is a diagnostic requirement here, not a newly fitted tolerance.
    Passing it alone does not certify hidden solver state or future dynamics.
    """
    errors = {}
    for key, value in source["proprioception"].items():
        errors[f"proprioception.{key}"] = array_error(
            value, candidate.get("proprioception", {}).get(key))
    sensor_metadata_match = True
    valid_sensors = True
    for group, sensors in source["images"].items():
        for sensor, fields in sensors.items():
            for field, value in fields.items():
                other = candidate.get("images", {}).get(group, {}).get(sensor, {}).get(field, {})
                errors[f"images.{group}.{sensor}.{field}"] = array_error(
                    value["data"], other.get("data"))
                valid_sensors &= bool(value.get("valid") and other.get("valid"))
                for key in ("sample_physics_step", "age_steps", "format"):
                    sensor_metadata_match &= value.get(key) == other.get(key)
    metadata = ("physics_step", "remaining_physics_steps", "remaining_recovery_steps",
                "previous_action", "history_mask", "references")
    metadata_match = {
        key: scene_digest(source.get(key)) == scene_digest(candidate.get(key))
        for key in metadata
    }
    measured_match = (all(e["compatible"] and e["max_abs"] == 0 for e in errors.values())
                      and sensor_metadata_match and valid_sensors and all(metadata_match.values()))
    return {"measurement": "exact_observation_probe.v1", "errors": errors,
            "metadata_match": metadata_match, "sensor_metadata_match": sensor_metadata_match,
            "valid_sensors": valid_sensors, "observations_match": bool(measured_match),
            "valid_match": False,
            "reason": "hidden_state_and_repeated_continuation_not_verified"}
