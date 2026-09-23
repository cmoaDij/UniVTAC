"""Contracts for training provenance, frozen evaluation, and durable recovery state."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evotac.config import ROOT, load_config, output_path
from evotac.data.legacy_demonstrations import LegacyDemonstrations, load_legacy_config
from evotac.learning.recovery_buffer import Transition


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_splits(config_path):
    config = load_config(config_path)
    legacy = LegacyDemonstrations(load_legacy_config(ROOT / config['legacy_data']['config']))
    result = {row['parent_scene_id']: row['split'] for row in legacy.manifest()}
    path = output_path(config['logging']['dataset_root']) / 'parent_splits.json'
    if path.exists():
        for parent, split in json.loads(path.read_text()).items():
            if parent in result and result[parent] != split:
                raise ValueError('scene registry conflicts with legacy source')
            result[parent] = split
    return result


def resolved_split(seed, splits):
    parent = f'insert_hole:source_seed:{seed}'
    fraction = int(hashlib.sha256(parent.encode()).hexdigest()[:8], 16) / 2**32
    return splits.get(parent, 'train' if fraction < .8 else 'dev' if fraction < .9 else 'test')


def validate_parents(seeds, splits, mode, requested_split=None):
    assigned = [resolved_split(seed, splits) for seed in seeds]
    if requested_split is not None and any(split != requested_split for split in assigned):
        raise ValueError(f'seed splits {assigned} do not match requested {requested_split}')
    if mode == 'train' and any(split != 'train' for split in assigned):
        raise ValueError('training must only use train parents; use --mode evaluate for dev/test')
    return assigned


def load_replay(path, buffer, warmstart):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if (meta.get('schema') != 'evotac.recovery_replay_dataset.v2'
                or meta.get('split') != 'train'
                or meta.get('warmstart_sha256') != sha256(warmstart)
                or not meta.get('parents') or any(p['split'] != 'train' for p in meta['parents'])):
            raise ValueError('replay schema, train provenance or frozen encoder mismatch')
        for i in range(len(data['reward'])):
            buffer.add(Transition(**{key: (bool(data[key][i]) if key in {
                'terminated', 'truncated', 'bootstrap_allowed'} else data[key][i])
                for key in ('observation', 'action', 'reward', 'next_observation',
                            'terminated', 'truncated', 'bootstrap_allowed')}))
    return meta


def checkpoint_contract(warmstart, skill_name, controls):
    return {'warmstart_sha256': sha256(warmstart), 'skill_name': skill_name,
            'monitor_object_lost_risk': controls['monitor_object_lost_risk'],
            'monitor_contact_blocked': controls['monitor_contact_blocked'],
            'trigger_calibration_sha256': controls.get('trigger_calibration_sha256'),
            'stable_cycles': controls['stable_cycles'],
            'max_recovery_actions': int(controls.get('max_recovery_actions', 0))}


def load_training_checkpoint(path, trainer, contract, *, evaluation=False):
    state = torch.load(path, map_location=trainer.learner.device, weights_only=False)
    evidence = state.get('training_evidence', {})
    if (state.get('experiment_contract', {}).get('warmstart_sha256') != contract['warmstart_sha256']
            or state.get('experiment_contract', {}).get('skill_name') != contract['skill_name']
            or evidence.get('split') != 'train'):
        raise ValueError('unverified training checkpoint or frozen encoder mismatch')
    if evaluation:
        trainer.learner.load_state_dict(state['learner'])
    else:
        saved_contract = dict(state['experiment_contract'])
        # Checkpoints created before the explicit action-cap field used the
        # config-derived cap. Treat the missing field as that legacy default.
        saved_contract.setdefault('max_recovery_actions', 0)
        if saved_contract != contract:
            raise ValueError('resume training must preserve trigger and handoff contract')
        trainer.load_state_dict(state)
        if 'torch_rng' in state:
            torch.set_rng_state(state['torch_rng'].cpu())
    return evidence


def save_training_checkpoint(path, trainer, contract, evidence):
    state = trainer.state_dict()
    state.update(experiment_contract=contract, training_evidence=evidence,
                 torch_rng=torch.get_rng_state())
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def evaluate_episode(driver, learner, seed, **kwargs):
    """Evaluation executes the driver without any learner update or trainer call."""
    before = {key: value.detach().clone() for key, value in learner.actor.state_dict().items()}
    result = driver.run(seed, **kwargs)
    if any(not torch.equal(value, learner.actor.state_dict()[key]) for key, value in before.items()):
        raise RuntimeError('actor changed during evaluation')
    return {'status': result.status, 'reason': result.reason,
            'recovery_actions': result.recovery_actions, 'admitted_transitions': result.admitted_transitions,
            'terminal_outcome': result.terminal_outcome, 'updates': [], 'skipped_updates': 0,
            'evaluation_weights_unchanged': True}
