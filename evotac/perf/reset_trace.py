"""Opt-in offline UIPC reset diagnostics; never included in policy observations."""
from functools import wraps

import h5py
import numpy as np

from evotac.data.rollout_logger import write_tree
from evotac.data.schemas import freeze


class ResetTrace:
    """Read backend positions/velocities around recovery and initialization.

    The state accessor only copies from the solver to a host scratch geometry.
    No rendering, stepping, state writes, or random draws are performed here.
    """

    def __init__(self, task, path):
        self.task = task
        self.handle = h5py.File(path, "x")
        self.attempt = -1
        self.originals = []
        self._patch(task.uipc_sim, "replay_frame", self._recover)
        self._patch(task, "_reset_actors", self._actors)
        self._patch(task, "_step", self._step)
        self._patch(task, "_stabilize_and_calibrate_marker_references", self._calibrate)
        self._patch(task, "pre_move", self._prefix)

    def _patch(self, owner, name, callback):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        @wraps(original)
        def wrapped(*args, **kwargs):
            return callback(original, *args, **kwargs)

        setattr(owner, name, wrapped)

    def _recover(self, original, *args, **kwargs):
        self.attempt += 1
        result = original(*args, **kwargs)
        self.capture("recovered")
        return result

    def _actors(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if self.attempt >= 0:
            self.capture("actors_reset")
        return result

    def _step(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        step = self.task._physics_step_count
        if self.attempt >= 0 and step in (1, 2, 5, 20, 25, 27, 37, 348, 523, 555, 627):
            self.capture(f"step_{step}")
        return result

    def _calibrate(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if self.attempt >= 0:
            self.capture("calibrated")
        return result

    def _prefix(self, original, *args, **kwargs):
        if self.attempt >= 0:
            self.capture("prefix_start")
        result = original(*args, **kwargs)
        if self.attempt >= 0:
            self.capture("prefix_end")
        return result

    def capture(self, stage):
        from uipc import builtin

        states = {}
        for obj in self.task.uipc_sim.uipc_objects:
            # Accessors have separate schemas for affine bodies and FEM nodes.
            obj._state_accessor.copy_to(obj._state_geo)
            geo = obj._state_geo
            rigid = hasattr(obj, "backend_system_body_offset")
            attributes = geo.instances() if rigid else geo.vertices()
            position = geo.transforms() if rigid else attributes.find(builtin.position)
            states[obj.cfg.prim_path.rsplit("/", 1)[-1]] = {
                "position": np.asarray(position.view()).copy(),
                "velocity": np.asarray(attributes.find(builtin.velocity).view()).copy(),
            }
        group = self.handle.require_group(str(self.attempt)).create_group(stage)
        write_tree(group, {"physics_step": int(self.task._physics_step_count),
                           "uipc_frame": int(self.task.uipc_sim.world.frame()),
                           "robot": freeze(self.task.read_robot_state().as_dict()),
                           "objects": states})
        self.handle.flush()

    def close(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
        self.handle.close()
