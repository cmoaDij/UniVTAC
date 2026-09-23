import numpy as np

from evotac.learning.actor_drift import build_probe_states, _pair_summary
from evotac.learning.recovery_sac import RecoverySAC


def test_probe_states_and_drift_are_deterministic():
    states = build_probe_states(count=8, seed=4)
    assert states.shape == (8, 128)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0)
    first = RecoverySAC(128, 7, hidden_dim=128, device="cpu", seed=2).actor.state_dict()
    second = {key: value.clone() for key, value in first.items()}
    summary = _pair_summary("warmstart", first, "candidate", second, states)
    assert summary["parameter_l2"] == 0.0
    assert summary["action_max_abs"] == 0.0
