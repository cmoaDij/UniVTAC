"""Read-only consistency audit of recorded control-rollout episodes."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import output_path
from evotac.data.rollout_logger import read_tree

DEPLOY_FIELDS = {"images", "proprioception", "proprioception_convention", "references", "physics_step",
                 "sim_time", "remaining_physics_steps", "previous_action", "history_mask"}
DEPLOY_FIELDS.add("remaining_recovery_steps")


def audit_episode(path, physical_hz=120, decimation=6):
    errors = []
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("data_kind") != "control_rollout":
            raise ValueError(f"Not a control rollout: {path}")
        counts = {name: len(handle[name]) for name in ("observations", "actions", "transitions")}
        finalized = bool(handle.attrs["complete"])
        if counts["actions"] != counts["transitions"]:
            errors.append("Action/transition count mismatch")
        for key in handle["transitions"]:
            transition = read_tree(handle["transitions"][key])
            if str(transition["action_index"]) not in handle["actions"]:
                errors.append(f"{key}: missing action")
                continue
            execution = read_tree(handle["actions"][str(transition["action_index"])])
            for index_field in ("observation_index", "next_observation_index"):
                index = transition[index_field]
                if index is not None and str(index) not in handle["observations"]:
                    errors.append(f"{key}: missing {index_field}")
            steps = execution["physics_steps"]
            if steps != execution["physics_end"]-execution["physics_start"]:
                errors.append(f"{key}: inconsistent physics count")
            if not np.isclose(execution["sim_time_end"]-execution["sim_time_start"], steps/physical_hz, rtol=0, atol=1e-10):
                errors.append(f"{key}: inconsistent simulation time")
            if bool(transition["is_executed_transition"]) != (steps > 0):
                errors.append(f"{key}: fabricated/missing executed-transition flag")
            applied = execution["applied_action"]
            if execution["status"] == "executed" and (steps != decimation or applied["accepted"] is not True):
                errors.append(f"{key}: normal action did not satisfy execution contract")
            if applied["accepted"] is True and applied.get("force") is not False:
                errors.append(f"{key}: accepted command lacks force=False")
            if transition["truncated"] and transition["bootstrap_allowed"]:
                errors.append(f"{key}: invalid finite-budget/infrastructure bootstrap")
        for key in handle["observations"]:
            obs = read_tree(handle["observations"][key])
            if set(obs) != DEPLOY_FIELDS:
                errors.append(f"observation {key}: deployment whitelist mismatch")
            for group in obs["images"].values():
                for sensor in group.values():
                    for modality in sensor.values():
                        sample = modality["sample_physics_step"]
                        if sample is not None and (sample > obs["physics_step"] or modality["age_steps"] != obs["physics_step"]-sample):
                            errors.append(f"observation {key}: incorrect modality age/time")
                        if sample is not None and not np.isclose(modality["sample_time"], sample/physical_hz):
                            errors.append(f"observation {key}: incorrect modality timestamp")
        outcome = json.loads(handle.attrs["outcome"]) if "outcome" in handle.attrs else None
        if finalized and outcome is None:
            errors.append("Finalized episode lacks outcome")
        return {"file": str(path), "finalized": finalized, "counts": counts, "outcome": outcome,
                "status": "failed" if errors else "consistent", "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit_episode(p) for p in sorted((args.dataset_run / "episodes").glob("*.h5"))]
    if not rows:
        raise ValueError("No control episodes found")
    result = {"status": "failed" if any(row["errors"] for row in rows) else "consistent", "episodes": rows}
    destination = output_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(result["status"], len(rows), "episodes")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
