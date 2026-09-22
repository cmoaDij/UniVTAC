"""Read-only structural audit of legacy UniVTAC HDF5 demonstrations.

Inspect every episode's headers and small state arrays. Decode first/middle/last
images and depth frames in three representative episodes per task. This does
not check every image payload or establish simulator compatibility.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def audit_task(task_dir):
    metadata_path = task_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    files = sorted((task_dir / "hdf5").glob("*.hdf5"), key=lambda p: int(p.stem))
    sample_ids = {0, len(files) // 2, len(files) - 1}
    episodes = []
    image_checks = []
    schemas = Counter()
    all_steps = Counter()
    errors = []
    for index, path in enumerate(files):
        meta = metadata.get(path.stem, {})
        row = {
            "file": str(path.resolve()), "episode_id": path.stem,
            "size_bytes": path.stat().st_size,
            "source_seed": meta.get("source_seed"), "seed": meta.get("seed"),
            "result": meta.get("result"), "metadata": meta,
        }
        try:
            with h5py.File(path, "r") as handle:
                datasets = {}
                attributes = {}

                def visit(name, obj):
                    if obj.attrs:
                        attributes[name] = list(obj.attrs)
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

                handle.visititems(visit)
                row["attribute_keys"] = {"root": list(handle.attrs), **attributes}
                schema = tuple(sorted(datasets))
                schemas[schema] += 1
                row["datasets"] = datasets
                steps = handle["step"][:]
                row["frames"] = len(steps)
                row["first_step"] = int(steps[0])
                row["last_step"] = int(steps[-1])
                differences = Counter(map(int, np.diff(steps)))
                all_steps.update(differences)
                row["step_differences"] = dict(differences)
                row["length_mismatches"] = [
                    name for name, spec in datasets.items()
                    if not spec["shape"] or spec["shape"][0] != len(steps)
                ]
                row["nonfinite_state_fields"] = []
                for name in datasets:
                    if name.startswith(("actor/", "embodiment/")):
                        if not np.isfinite(handle[name][:]).all():
                            row["nonfinite_state_fields"].append(name)
                joints = handle["embodiment/joint"][:]
                row["joint_min"] = joints.min(axis=0).tolist()
                row["joint_max"] = joints.max(axis=0).tolist()
                row["max_finger_difference"] = float(np.max(np.abs(joints[:, -1] - joints[:, -2])))
                row["atom_tags"] = sorted({v.decode() for v in handle["atom/tag"][:]})
                row["atom_ids"] = sorted(set(map(int, handle["atom/id"][:])))
                if index in sample_ids:
                    for name in datasets:
                        if "rgb" in name:
                            for frame in sorted({0, len(steps) // 2, len(steps) - 1}):
                                raw = handle[name][frame]
                                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if image is None:
                                    raise ValueError(f"Image decode failed: {name}[{frame}]")
                                image_checks.append({
                                    "episode_id": path.stem, "field": name, "frame": frame,
                                    "shape": list(image.shape), "std": float(image.std()),
                                    "min": int(image.min()), "max": int(image.max()),
                                })
                        elif name.endswith(("/depth", "/press_depth", "/marker")):
                            for frame in sorted({0, len(steps) // 2, len(steps) - 1}):
                                if not np.isfinite(handle[name][frame]).all():
                                    errors.append(f"{path.name}: nonfinite {name}[{frame}]")
                if any(delta <= 0 for delta in differences):
                    errors.append(f"{path.name}: non-increasing step")
                if row["length_mismatches"] or row["nonfinite_state_fields"]:
                    errors.append(f"{path.name}: invalid lengths or state values")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{path.name}: {row['error']}")
        episodes.append(row)
    frames = [row["frames"] for row in episodes if "frames" in row]
    source_seeds = [row["source_seed"] for row in episodes if row["source_seed"] is not None]
    return {
        "task": task_dir.name, "episodes": episodes, "file_count": len(files),
        "size_gib": sum(row["size_bytes"] for row in episodes) / 2**30,
        "frames_total": sum(frames),
        "frames_min_median_max": [min(frames), float(np.median(frames)), max(frames)] if frames else [],
        "step_differences": dict(all_steps),
        "results": dict(Counter(row["result"] for row in episodes)),
        "metadata_count": len(metadata),
        "missing_metadata": [row["episode_id"] for row in episodes if row["episode_id"] not in metadata],
        "source_seed_range": [min(source_seeds), max(source_seeds)] if source_seeds else [],
        "duplicate_source_seeds": [seed for seed, n in Counter(source_seeds).items() if n > 1],
        "renumbered_episodes": sum(str(row["source_seed"]) != row["episode_id"] for row in episodes),
        "schema_variants": [{"count": n, "keys": list(keys)} for keys, n in schemas.items()],
        "sampled_image_checks": image_checks, "errors": errors,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root.parent / "data/isaac51")
    parser.add_argument("--output", type=Path, default=root / "runs/legacy_audit/isaac51_audit.json")
    args = parser.parse_args()
    report = {"source": str(args.source.resolve()), "scope": __doc__, "tasks": []}
    for task_dir in sorted(args.source.iterdir()):
        if not (task_dir / "hdf5").is_dir():
            continue
        result = audit_task(task_dir)
        report["tasks"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in {
            "episodes", "schema_variants", "sampled_image_checks"
        }}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
