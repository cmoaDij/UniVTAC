"""Read-only legacy access. Future measured qpos is never an accepted action."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from evotac.config import ROOT, output_path
from evotac.data.schemas import json_value


def load_legacy_config(path=None):
    return yaml.safe_load((Path(path) if path else ROOT / "configs/legacy_insert_hole.yaml").read_text())


def pair_indices(length, source_rate=60, target_rate=20, phase=0):
    if source_rate <= 0 or target_rate <= 0 or source_rate % target_rate:
        raise ValueError("Source rate must be a positive integer multiple of target rate")
    stride = source_rate // target_rate
    if type(phase) is not int or not 0 <= phase < stride:
        raise ValueError("Invalid sampling phase")
    inputs = np.arange(phase, max(phase, length-stride), stride, dtype=np.int64)
    return inputs, inputs + stride


def map_joints(joints, config):
    names = config["source_joint_names"]
    if len(set(names)) != len(names) or joints.shape[-1] != len(names):
        raise ValueError("Invalid source joint name mapping")
    arms = [names.index(n) for n in config["arm_joint_names"]]
    fingers = [names.index(n) for n in config["finger_joint_names"]]
    return np.concatenate([joints[..., arms], joints[..., fingers].mean(axis=-1, keepdims=True)], axis=-1)


def assign_splits(rows, ratios=(0.8, 0.1, 0.1), seed=20260917):
    """Assign complete parents before sampling. No child phase can change split."""
    if len(ratios) != 3 or min(ratios) < 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Expected three split ratios summing to one")
    parents = sorted({r["parent_scene_id"] for r in rows}, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
    train_end = int(len(parents)*ratios[0])
    dev_end = train_end + int(len(parents)*ratios[1])
    mapping = {p: "train" if i < train_end else "dev" if i < dev_end else "test" for i, p in enumerate(parents)}
    return [{**r, "split": mapping[r["parent_scene_id"]]} for r in rows]


class LegacyDemonstrations:
    def __init__(self, config=None):
        self.config = config or load_legacy_config()
        self.root = (ROOT / self.config["source_root"]).resolve()
        self.metadata = json.loads((self.root / "metadata.json").read_text())

    def manifest(self):
        rows = []
        c = self.config
        for path in sorted((self.root / "hdf5").glob("*.hdf5"), key=lambda p: int(p.stem)):
            meta = self.metadata.get(path.stem, {})
            # Never infer source seed from a renamed file number.
            seed = meta.get(c["source_seed_field"])
            parent = f"insert_hole:source_seed:{seed}" if seed is not None else f"insert_hole:unknown_seed:episode:{path.stem}"
            rows.append({"source_path": str(path), "source_episode_id": path.stem,
                         "source_seed": seed, "parent_scene_id": parent,
                         "source_version": c["source_version"], "data_kind": "legacy_demonstration",
                         "source_rate_evidence": c["source_rate_evidence"],
                         "source_rate_confirmed": c["source_rate_confirmed"],
                         "conversion_version": c["conversion_version"], "action_semantics": c["action_semantics"],
                         "source_size_bytes": path.stat().st_size,
                         "training_eligible": bool(c["source_rate_confirmed"] and c["joint_order_confirmed"]),
                         "missing": ["applied_action", "references", "sensor_timestamps", "prefix_commands"],
                         "image_source_encoding": "lossy_jpeg"})
        return assign_splits(rows, c["split_ratios"], c["split_seed"])

    def candidate(self, row, phase=0):
        c = self.config
        with h5py.File(row["source_path"], "r") as handle:
            joints = map_joints(handle["embodiment/joint"][:], c)
            inputs, targets = pair_indices(len(joints), c["source_rate_hz"], c["target_rate_hz"], phase)
            return {"input_indices": inputs, "target_indices": targets,
                    "joint_position": joints[inputs], "future_measured_qpos": joints[targets],
                    "source_step": handle["step"][:][inputs],
                    "target_horizon_source_frames": c["source_rate_hz"]//c["target_rate_hz"],
                    "target_horizon_seconds": 1/c["target_rate_hz"] if c["source_rate_confirmed"] else None,
                    "hypothesized_horizon_seconds": 1/c["target_rate_hz"], "phase": phase}

    def read_observation(self, row, index):
        """Decode one source frame using the original codec's channel round trip.

        The legacy writer passed its image directly to cv2.imencode and the
        reader uses imdecode without a channel swap. Preserve that convention;
        historical color-order confirmation remains a phase-2 provenance gate.
        """
        import cv2
        with h5py.File(row["source_path"], "r") as handle:
            if not 0 <= index < len(handle["step"]):
                raise IndexError("Legacy frame is outside the source episode")
            def decode(key):
                payload = handle[key][index]
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot decode {key} at source frame {index}")
                return image
            return {"camera": {target: {"rgb": decode(f"observation/{source}/rgb")}
                               for source, target in self.config["camera_mapping"].items()},
                    "tactile": {name: {field: decode(f"tactile/{name}/{field}") for field in ("rgb", "rgb_marker")}
                                for name in ("left_tactile", "right_tactile")},
                    "joint_position": map_joints(handle["embodiment/joint"][index], self.config),
                    "source_frame_index": int(index), "sensor_timestamps": None,
                    "references": None, "applied_action": None,
                    "source_image_encoding": "lossy_jpeg",
                    "channel_convention": "legacy_opencv_roundtrip_without_swap"}

    def convert_sample(self, destination, count=3):
        """Store low-dimensional pairs and image references without copying JPEGs."""
        destination = output_path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "episodes").mkdir()
        rows = self.manifest()
        selected = [r for r in rows if r["split"] in ("train", "dev")][:count]
        from evotac.data.rollout_logger import write_tree
        for row in selected:
            with h5py.File(destination / "episodes" / f'{row["source_episode_id"]}.h5', "x") as handle:
                candidate = self.candidate(row)
                write_tree(handle, candidate)
                if len(candidate["input_indices"]):
                    write_tree(handle.create_group("sample_observation"), self.read_observation(row, int(candidate["input_indices"][0])))
                handle.attrs["source"] = json.dumps(row)
                handle.attrs["images"] = "Read source JPEGs at input_indices; sensor timestamps unknown"
        manifest = {"data_kind": "legacy_demonstration", "config": self.config, "episodes": rows,
                    "converted_ids": [r["source_episode_id"] for r in selected],
                    "compatibility": "insufficient_evidence_pending_gpu_tracking"}
        (destination / "manifest.json").write_text(json.dumps(json_value(manifest), indent=2))
        return manifest
