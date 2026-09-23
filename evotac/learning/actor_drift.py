"""CPU diagnostics for warm-start retention and actor policy drift.

This module compares frozen actor checkpoints on a deterministic probe set.  It
is a reproducible parameter/action diagnostic; it does not stand in for an
Isaac rollout or a causal efficacy result.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from evotac.learning.recovery_sac import _GaussianActor


OBSERVATION_DIM = 128
ACTION_DIM = 7
HIDDEN_DIM = 128


def build_probe_states(count=256, seed=20260923, observation_dim=OBSERVATION_DIM):
    if type(count) is not int or count < 1:
        raise ValueError("probe count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("probe seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((count, observation_dim), dtype=np.float32)
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    return states / np.maximum(norms, np.finfo(np.float32).eps)


def _checkpoint_actor(checkpoint: Path, *, kind: str):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if kind == "trained":
        if state.get("schema") != "evotac.recovery_sac_trainer.v1":
            raise ValueError("trained checkpoint is not a recovery trainer checkpoint")
        actor = state.get("learner", {}).get("actor")
        network = state.get("learner", {}).get("network_config", {})
        skill = state.get("experiment_contract", {}).get("skill_name")
    elif kind == "warmstart":
        if state.get("schema") != "evotac.control_warmstart.v1":
            raise ValueError("warm-start checkpoint is not a control warm-start")
        skills = state.get("skill_names", ())
        if len(skills) != 1:
            raise ValueError("actor drift requires a single-skill warm-start checkpoint")
        skill = skills[0]
        actor = {key.removeprefix(f"{skill}."): value
                 for key, value in state.get("actors", {}).items()
                 if key.startswith(f"{skill}.")}
        network = {"observation_dim": OBSERVATION_DIM, "action_dim": ACTION_DIM,
                   "hidden_dim": HIDDEN_DIM}
    else:
        raise ValueError(f"unsupported actor checkpoint kind: {kind}")
    if not actor or network.get("observation_dim") != OBSERVATION_DIM or network.get("action_dim") != ACTION_DIM:
        raise ValueError("actor checkpoint does not match the frozen 128D/7D contract")
    return {key: value.detach().cpu().clone() for key, value in actor.items()}, {
        "path": str(Path(checkpoint).resolve()),
        "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "kind": kind, "skill_name": skill,
    }


def _action_matrix(actor_state, states):
    actor = _GaussianActor(OBSERVATION_DIM, ACTION_DIM, HIDDEN_DIM)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    values = torch.as_tensor(np.asarray(states, dtype=np.float32))
    with torch.inference_mode():
        actions, _ = actor.sample(values, deterministic=True)
    return actions.cpu().numpy()


def _pair_summary(reference_name, reference_state, candidate_name, candidate_state, states):
    keys = sorted(reference_state)
    parameter_delta = torch.cat([(candidate_state[key] - reference_state[key]).reshape(-1)
                                 for key in keys]).numpy()
    reference_actions = _action_matrix(reference_state, states)
    candidate_actions = _action_matrix(candidate_state, states)
    action_delta = candidate_actions - reference_actions
    return {
        "reference": reference_name, "candidate": candidate_name,
        "parameter_l2": float(np.linalg.norm(parameter_delta)),
        "parameter_max_abs": float(np.max(np.abs(parameter_delta))),
        "parameter_relative_l2": float(np.linalg.norm(parameter_delta) /
                                         max(np.linalg.norm(torch.cat([reference_state[k].reshape(-1) for k in keys]).numpy()), 1e-12)),
        "action_rmse": float(np.sqrt(np.mean(action_delta ** 2))),
        "action_mean_abs": float(np.mean(np.abs(action_delta))),
        "action_max_abs": float(np.max(np.abs(action_delta))),
        "reference_action_saturation_rate": float(np.mean(np.abs(reference_actions) >= 0.99)),
        "candidate_action_saturation_rate": float(np.mean(np.abs(candidate_actions) >= 0.99)),
    }


def audit_actor_drift(trained_checkpoint, warmstart_checkpoint, *, candidate_checkpoint=None,
                      count=256, seed=20260923):
    states = build_probe_states(count=count, seed=seed)
    trained, trained_meta = _checkpoint_actor(Path(trained_checkpoint), kind="trained")
    warm, warm_meta = _checkpoint_actor(Path(warmstart_checkpoint), kind="warmstart")
    if set(trained) != set(warm):
        raise ValueError("trained and warm-start actor keys do not match")
    actors = {"warmstart": warm, "trained": trained}
    metadata = {"warmstart": warm_meta, "trained": trained_meta}
    if candidate_checkpoint is not None:
        candidate, candidate_meta = _checkpoint_actor(Path(candidate_checkpoint), kind="trained")
        if set(candidate) != set(warm):
            raise ValueError("candidate and warm-start actor keys do not match")
        actors["candidate"] = candidate
        metadata["candidate"] = candidate_meta
    pairs = [_pair_summary("warmstart", warm, "trained", trained, states)]
    if "candidate" in actors:
        pairs.extend([_pair_summary("warmstart", warm, "candidate", actors["candidate"], states),
                      _pair_summary("trained", trained, "candidate", actors["candidate"], states)])
    return {
        "schema": "evotac.actor_drift_audit.v1",
        "evidence_scope": "cpu deterministic probe diagnostic; not physical causal evidence",
        "probe": {"count": count, "seed": seed, "observation_dim": OBSERVATION_DIM,
                  "construction": "unit-normalized standard-normal states"},
        "actors": metadata, "pairs": pairs,
    }
