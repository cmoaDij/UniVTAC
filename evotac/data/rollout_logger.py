"""Append-only per-attempt HDF5, including failed reset and zero-step rejection."""
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import output_path
from evotac.data.schemas import freeze, json_value


def write_tree(group, values):
    for key, value in freeze(values).items():
        if "/" in key:
            raise ValueError("HDF5 record keys must not contain slash")
        if isinstance(value, dict):
            write_tree(group.create_group(key), value)
        elif value is None:
            group.create_group(key).attrs["is_none"] = True
        elif isinstance(value, str):
            group.create_dataset(key, data=value, dtype=h5py.string_dtype())
        elif isinstance(value, (list, tuple)) and (not value or any(isinstance(v, (dict, str)) or v is None for v in value)):
            sequence = group.create_group(key)
            sequence.attrs["sequence"] = True
            for index, item in enumerate(value):
                write_tree(sequence, {str(index): item})
        else:
            array = np.asarray(value)
            group.create_dataset(key, data=array, **({"compression": "gzip", "shuffle": True} if array.ndim and array.size else {}))


def read_tree(group):
    if group.attrs.get("sequence", False):
        return [read_tree_item(group[str(i)]) for i in range(len(group))]
    result = {}
    for key, value in group.items():
        result[key] = read_tree_item(value)
    return result


def read_tree_item(value):
    if isinstance(value, h5py.Group):
        return None if value.attrs.get("is_none", False) else read_tree(value)
    item = value[()]
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, bytes):
        item = item.decode()
    if value.attrs.get("encoding") == "json":
        item = json.loads(item)
    return item


class RolloutLogger:
    def __init__(self, path, scene_metadata, flush_every=1):
        self.path = output_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "x")
        self.handle.attrs.update(complete=False, data_kind="control_rollout",
                                 scene_metadata=json.dumps(json_value(scene_metadata)))
        for name in ("observations", "actions", "transitions", "training_info", "references", "events"):
            self.handle.create_group(name)
        self.count, self.observation_count, self.event_count = 0, 0, 0
        self.flush_every = flush_every
        self.handle.flush()  # durable attempt record before reset

    def initial_observation(self, observation, references):
        if self.observation_count:
            raise RuntimeError("Initial observation already written")
        write_tree(self.handle["observations"].create_group("0"), observation)
        write_tree(self.handle["references"], references)
        self.observation_count = 1
        self.handle.flush()

    def append(self, execution, next_observation, outcome, training_info):
        if self.observation_count != self.count + 1:
            raise RuntimeError("Missing pre-action observation")
        index = str(self.count)
        write_tree(self.handle["actions"].create_group(index), execution)
        next_index = None
        if next_observation is not None:
            next_index = self.observation_count
            write_tree(self.handle["observations"].create_group(str(next_index)), next_observation)
            self.observation_count += 1
        write_tree(self.handle["training_info"].create_group(index), training_info)
        write_tree(self.handle["transitions"].create_group(index), {
            "observation_index": self.count, "next_observation_index": next_index,
            "action_index": self.count, "is_executed_transition": execution["physics_steps"] > 0,
            **outcome})
        self.count += 1
        if self.count % self.flush_every == 0:
            self.handle.flush()

    def event(self, kind, details):
        write_tree(self.handle["events"].create_group(str(self.event_count)), {"kind": kind, **details})
        self.event_count += 1
        self.handle.flush()

    def finish(self, outcome):
        self.handle.attrs["outcome"] = json.dumps(json_value(outcome))
        self.handle.attrs["complete"] = True
        self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.flush()
            self.handle.close()
