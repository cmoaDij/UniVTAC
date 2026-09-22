"""Optional CPU wall timers; no added simulator ticks or GPU synchronizations."""
from contextlib import contextmanager
from functools import wraps
import time


class ScopedTiming:
    """Temporarily wrap live instance methods and separate timed phases.

    Inclusive durations overlap (render contains tactile updates). Exclusive
    durations subtract timed children in the same thread. These are CPU wall
    times including existing synchronization, not CUDA kernel durations.
    """

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.phase = "setup"
        self.rows = {}
        self.stack = []
        self.originals = []
        self.unavailable = {}

    @contextmanager
    def measure(self, name):
        start = self.clock()
        frame = [0.0]
        phase = self.phase
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - start
            self.stack.pop()
            if self.stack:
                self.stack[-1][0] += elapsed
            row = self.rows.setdefault((phase, name), {"calls": 0, "inclusive_seconds": 0.0,
                                                      "exclusive_seconds": 0.0, "max_seconds": 0.0})
            row["calls"] += 1
            row["inclusive_seconds"] += elapsed
            row["exclusive_seconds"] += max(0.0, elapsed - frame[0])
            row["max_seconds"] = max(row["max_seconds"], elapsed)

    def wrap(self, obj, method, name):
        try:
            original = getattr(obj, method)
        except AttributeError:
            self.unavailable[name] = "missing_method"
            return False
        try:
            owned = method in vars(obj)
        except TypeError:
            owned = False

        @wraps(original)
        def measured(*args, **kwargs):
            with self.measure(name):
                return original(*args, **kwargs)

        try:
            setattr(obj, method, measured)
        except (AttributeError, TypeError):
            self.unavailable[name] = "method_not_assignable"
            return False
        self.originals.append((obj, method, original, owned))
        return True

    def close(self):
        for obj, method, original, owned in reversed(self.originals):
            if owned:
                setattr(obj, method, original)
            else:
                delattr(obj, method)
        self.originals.clear()

    def report(self):
        out = {}
        for (phase, name), row in sorted(self.rows.items()):
            out.setdefault(phase, {})[name] = dict(row)
        if self.unavailable:
            out["_unavailable"] = dict(self.unavailable)
        return out


def instrument_wrapper(wrapper, timing):
    task = wrapper.task
    for obj, method, label in (
        (wrapper, "_observe", "observation"), (wrapper, "step", "wrapper_step"),
        (wrapper.executor, "execute", "executor"), (task, "_update_render", "render_and_sensors"),
        (task.sim, "step", "physics_step"), (task.sim, "render", "renderer"),
        (task.uipc_sim, "update_render_meshes", "uipc_mesh"),
        (task.scene, "update", "scene_update"),
        (task._actor_manager, "update", "actor_update"),
        (task._tactile_manager, "update", "tactile_update"),
        (wrapper.contract, "accept", "history_append"),
        (wrapper.policy_state, "rebuild", "history_copy"),
    ):
        timing.wrap(obj, method, label)
    # UIPC physics is invoked by a previously registered bound callback. A
    # wrapper on task.uipc_sim.step now would miss it, so physics_step includes
    # its wall time and we deliberately do not claim separate kernel timing.
