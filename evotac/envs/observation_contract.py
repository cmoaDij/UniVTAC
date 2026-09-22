"""Build deployment inputs by whitelist; privileged state has a separate channel."""
from collections import deque
import numpy as np

from evotac.data.schemas import freeze, ARM_NAMES, FINGER_NAMES


class ObservationContract:
    def __init__(self, config, physical_hz=120):
        self.config, self.physical_hz = config, physical_hz
        self.history = deque(maxlen=config["history_length"])
        self.previous_action = None

    def reset(self):
        self.history.clear()
        self.previous_action = None

    def build(self, raw, state, physics_step, sample_steps, references, remaining_steps, sample_frames=None):
        for name, size in (("joint_position", 7), ("joint_velocity", 7), ("finger_position", 2),
                           ("ee_position_world", 3), ("ee_quaternion_world", 4), ("base_quaternion_world", 4)):
            value = np.asarray(getattr(state, name))
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid proprioception: {name}")
            if "quaternion" in name and not np.isclose(np.linalg.norm(value), 1.0, atol=1e-3):
                raise ValueError(f"Invalid proprioception quaternion: {name}")
        images = {}
        sample_frames = sample_frames or {}
        for category, names, fields, raw_key in (
            ("camera", self.config["cameras"], ["rgb"], "observation"),
            ("tactile", self.config["tactile_sensors"], self.config["tactile_fields"], "tactile"),
        ):
            images[category] = {}
            for name in names:
                modalities = {}
                for field in fields:
                    image = raw.get(raw_key, {}).get(name, {}).get(field)
                    sampled = sample_steps.get(f"{category}/{name}/{field}")
                    if sampled is not None and sampled > physics_step:
                        raise ValueError("Sensor sample is in the future")
                    age = None if sampled is None else physics_step - sampled
                    data = freeze(image)
                    valid = (data is not None and sampled is not None and age >= 0
                             and age <= self.config["max_frame_age_steps"]
                             and np.asarray(data).ndim == 3
                             and np.asarray(data).shape[-1] == 3
                             and np.asarray(data).dtype == np.uint8)
                    modalities[field] = {"data": data, "sample_physics_step": sampled,
                                         "frame_id": sample_frames.get(f"{category}/{name}/{field}"),
                                         "sample_time": None if sampled is None else sampled/self.physical_hz,
                                         "age_steps": age, "valid": bool(valid), "format": "RGB_uint8_HWC",
                                         "contact_state": "unknown"}
                images[category][name] = modalities
        # References are produced by this contract, never arbitrary task metadata.
        safe_refs = {}
        for kind in ("empty", "grasp"):
            ref = references.get(kind)
            safe_refs[kind] = None if ref is None else freeze({
                key: ref[key] for key in ("images", "physics_step", "valid", "kind")})
        obs = {"images": images, "proprioception": state.as_dict(),
               "proprioception_convention": {"arm_joint_names": list(ARM_NAMES), "finger_joint_names": list(FINGER_NAMES),
                                              "arm_position_unit": "rad", "arm_velocity_unit": "rad/s",
                                              "finger_position_unit": "m", "ee_position_unit": "m",
                                              "ee_frame": "world", "quaternion_order": "wxyz", "control_point": "panda_hand"},
               "references": safe_refs, "physics_step": int(physics_step),
               "sim_time": physics_step/self.physical_hz, "remaining_physics_steps": int(remaining_steps),
               "previous_action": freeze(self.previous_action),
               "history_mask": [False] * (self.history.maxlen-len(self.history)) + [True] * len(self.history)}
        return freeze(obs)

    def accept(self, obs, action=None):
        self.history.append({"observation": freeze(obs), "action": freeze(action)})
