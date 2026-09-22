"""Persistent parent-scene split assignments, shared by runs and perturbations."""
import fcntl
import hashlib
import json
import os
import uuid

from evotac.config import output_path


class SceneRegistry:
    def __init__(self, root, initial_splits=None):
        self.root = output_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "parent_splits.json"
        self.lock_path = self.root / ".parent_splits.lock"
        self.initial_splits = dict(initial_splits or {})

    def register(self, parent, requested_split=None):
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            splits = json.loads(self.path.read_text()) if self.path.exists() else {}
            for key, value in self.initial_splits.items():
                if key in splits and splits[key] != value:
                    raise ValueError(f"Existing split conflicts with source provenance: {key}")
                splits[key] = value
            if requested_split is None:
                # Existing source assignments take precedence. For new parents,
                # a stable hash selects a split before any trajectories are sampled.
                fraction = int(hashlib.sha256(parent.encode()).hexdigest()[:8], 16) / 2**32
                requested_split = splits.get(parent, "train" if fraction < .8 else "dev" if fraction < .9 else "test")
            if requested_split not in {"train", "dev", "test"}:
                raise ValueError("Invalid parent split")
            if parent in splits and splits[parent] != requested_split:
                raise ValueError(f"Parent {parent} belongs to {splits[parent]}, not {requested_split}")
            splits[parent] = requested_split
            temporary = self.root / (".parent_splits_"+uuid.uuid4().hex+".tmp")
            try:
                temporary.write_text(json.dumps(splits, indent=2, sort_keys=True))
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return requested_split
