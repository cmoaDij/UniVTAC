"""Measure snapshot restoration; a shared identifier is never match evidence."""
from pathlib import Path
import json

from evotac.learning.recovery_experiment import sha256
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
    state_match = (all(
        errors[key]["compatible"] and errors[key]["max_abs"] <= 1e-6
        for key in errors if key.startswith("proprioception.")
    ) and sensor_metadata_match and valid_sensors and all(metadata_match.values()))
    measured_match = (state_match and all(e["compatible"] and e["max_abs"] == 0
                                          for e in errors.values()))
    image_errors = [e for key, e in errors.items() if key.startswith("images.")]
    images_compatible = bool(image_errors) and all(e["compatible"] for e in image_errors)
    image_max_abs = max((e["max_abs"] for e in image_errors), default=0.0) if images_compatible else None
    return {"measurement": "exact_observation_probe.v1", "errors": errors,
            "metadata_match": metadata_match, "sensor_metadata_match": sensor_metadata_match,
            "valid_sensors": valid_sensors, "state_match": bool(state_match),
            "image_max_abs": image_max_abs, "images_compatible": images_compatible,
            "observations_match": bool(measured_match),
            "valid_match": False,
            "reason": "hidden_state_and_repeated_continuation_not_verified"}


def continuation_audit(observations):
    """Compare four control restores and one candidate without relaxing gates.

    The empirical image range is diagnostic only. These restores share a
    reconstruction path, so their variation cannot isolate renderer noise
    from restoration drift or certify hidden state / continuous trajectories.
    """
    if len(observations) != 5:
        raise ValueError("four controls and one candidate are required")
    controls = [observation_audit(observations[left], observations[right])
                for left in range(4) for right in range(left + 1, 4)]
    candidate = observation_audit(observations[0], observations[4])
    compatible = all(row["images_compatible"] for row in [*controls, candidate])
    control_range = max(row["image_max_abs"] for row in controls) if compatible else None
    return {
        "measurement": "repeated_restore_diagnostic.v2",
        "controls": controls, "candidate": candidate,
        "control_image_max_abs": control_range,
        "within_control_image_range": bool(compatible and candidate["image_max_abs"] <= control_range),
        "state_match": all(row["state_match"] for row in [*controls, candidate]),
        "observations_match": all(row["observations_match"] for row in [*controls, candidate]),
        "valid_match": False,
        "reason": "hidden_state_and_repeated_continuation_not_verified",
    }


def strict_paired_valid(report):
    """Fail closed on incomplete branches, visible drift or uncertified state."""
    branches = report.get("branches", [])
    if len(branches) != 2 or {row.get("branch") for row in branches} != {"baseline", "recovery"}:
        return False
    probe = report.get("continuation_probe", {})
    if probe.get("observations_match") is not True or probe.get("valid_match") is not True:
        return False
    return all(
        row.get("status") == "finished"
        and row.get("reconstruction", {}).get("observations_match") is True
        and row.get("reconstruction", {}).get("valid_match") is True
        and row.get("reconstruction", {}).get("actor_state_match") is True
        and row.get("outcome", {}).get("valid_trial") is True
        and row.get("outcome", {}).get("incomplete") is False
        for row in branches
    )


def frozen_protocol(files, controls):
    """Fingerprint effective controls and files, never the evaluation parents."""
    return {"schema": "evotac.frozen_evaluation.v1",
            "files": {name: sha256(path) for name, path in sorted(files.items())},
            "controls": controls}


def load_test_gate(path, *, warmstart_sha256=None,
                   checkpoint_sha256=None, expected_protocol=None, test_seeds=()):
    """Load a completed dev snapshot audit before any independent test launch.

    The gate is deliberately stricter than a report's legacy ``paired_valid``
    flag: it re-evaluates the current fail-closed predicate and checks that the
    exact frozen actor inputs used by test were audited.  A gate
    from a test run is rejected to keep test outcomes from authorizing their
    own causal comparison.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"causal audit gate does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid causal audit gate JSON: {path}") from exc
    if report.get("audit_gate") != "strict_visible_and_hidden_state.v2":
        raise ValueError("causal audit gate schema is not strict_visible_and_hidden_state.v2")
    if report.get("split") not in {"dev", "acceptance"}:
        raise ValueError("independent test requires a strict gate produced on dev/acceptance")
    if report.get("test_started") is True:
        raise ValueError("a gate that already started test cannot authorize independent test")
    if not strict_paired_valid(report) or report.get("paired_valid") is not True:
        raise ValueError("strict snapshot audit is not valid; test remains blocked")
    if report.get("status") != "completed":
        raise ValueError("causal audit must be completed")
    if expected_protocol is None or report.get("frozen_protocol") != expected_protocol:
        raise ValueError("causal audit frozen protocol mismatch or missing protocol")
    if report.get("seed") in test_seeds:
        raise ValueError("test parents must be independent of dev audit parents")
    mismatches = []
    if warmstart_sha256 is not None and report.get("warmstart_sha256") != warmstart_sha256:
        mismatches.append("warmstart_sha256 mismatch")
    if checkpoint_sha256 is not None and report.get("checkpoint_sha256") != checkpoint_sha256:
        mismatches.append("checkpoint_sha256 mismatch")
    if mismatches:
        raise ValueError("causal audit gate frozen-input mismatch: " + "; ".join(mismatches))
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "audit_gate": report["audit_gate"], "split": report["split"],
            "source_seed": report.get("seed"), "warmstart_sha256": report.get("warmstart_sha256"),
            "checkpoint_sha256": report.get("checkpoint_sha256")}
