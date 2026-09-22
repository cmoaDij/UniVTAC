"""Re-encode recorded training recovery fragments with one frozen encoder.

Reconstruct the collector's delayed continuation labels from durable events;
never treat the logger's immediate handoff reward as a terminal recovery label.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from evotac.config import ROOT
from evotac.data.rollout_logger import read_tree
from evotac.learning.recovery_buffer import Transition
from evotac.learning.recovery_observation import RecoveryHistory
from evotac.learning.recovery_warmstart import RecoveryWarmStart
from evotac.policy.tactile_features import CachedTactileEncoder, FrozenTactileEncoder, recovery_features


def recovery_records(handle, discount=0.99):
    """Chronological actions and exact collector labels, without image encoding."""
    outcome = json.loads(handle.attrs.get('outcome', '{}'))
    if not handle.attrs.get('complete') or not outcome.get('valid_trial') or outcome.get('incomplete'):
        raise ValueError('only complete valid episodes can enter replay')
    records = []
    for key in sorted(handle['actions'], key=int):
        action = read_tree(handle['actions'][key])
        if action.get('proposed_action', {}).get('kind') != 'recovery_delta':
            continue
        transition = read_tree(handle['transitions'][key])
        if (action.get('status') != 'executed' or action.get('physics_steps') != 6
                or transition.get('valid_trial') is not True
                or transition.get('next_observation_index') is None):
            raise ValueError('invalid recovery fragment must not enter replay')
        records.append({'action_index': int(key), 'action': action['proposed_action']['values'],
                        **{name: transition[name] for name in (
                            'observation_index', 'next_observation_index', 'reward',
                            'terminated', 'truncated', 'bootstrap_allowed')}})
    if not records:
        return []
    events = [read_tree(handle['events'][key]) for key in sorted(handle['events'], key=int)]
    handoffs = [event for event in events if event.get('kind') == 'handoff']
    if handoffs:
        continuations = [event for event in events if event.get('kind') == 'continuation']
        if len(handoffs) != 1 or len(continuations) != 1:
            raise ValueError('expected one durable handoff and continuation')
        terminal = continuations[0]['outcome']
        if (terminal.get('reason') != outcome['reason'] or not terminal.get('valid_trial')
                or terminal.get('incomplete')
                or terminal['reason'] not in {'success', 'object_lost', 'task_budget'}
                or not (terminal.get('terminated') or terminal.get('truncated'))):
            raise ValueError('continuation is invalid or inconsistent with episode outcome')
        last = records[-1]
        last.update(reward=float(last['reward']) + discount * float(terminal['reason'] == 'success'),
                    terminated=bool(terminal['terminated']), truncated=bool(terminal['truncated']),
                    bootstrap_allowed=False)
    elif not (records[-1]['terminated'] or records[-1]['truncated']):
        raise ValueError('recovery fragment has neither terminal nor handoff evidence')
    return records


def extract(episodes, warmstart, *, split='train', skill_name='small_lift_adjust_reapproach',
            device='cpu'):
    model = RecoveryWarmStart.from_checkpoint(torch.load(warmstart, map_location='cpu', weights_only=False),
                                             device=device)
    if skill_name not in model.skill_names:
        raise ValueError('skill absent from warmstart')
    provenance = json.loads((ROOT / 'checkpoints/univtac_release/encoder_source.json').read_text())
    tactile = CachedTactileEncoder(FrozenTactileEncoder(
        ROOT / 'checkpoints/univtac_release/encoder.pth', provenance['sha256'], device=device))
    rows, parents, sources, seen = [], [], [], set()
    for episode_path in episodes:
        with h5py.File(episode_path, 'r') as handle:
            metadata = json.loads(handle.attrs['scene_metadata'])
            if metadata.get('split') != split:
                raise ValueError(f'split mismatch: {episode_path}')
            records = recovery_records(handle)
            if not records:
                continue
            parent = metadata.get('parent_scene_id')
            if not isinstance(parent, str) or parent in seen:
                raise ValueError(f'missing or duplicate parent: {parent}')
            seen.add(parent)
            needed = sorted({r[k] for r in records for k in ('observation_index', 'next_observation_index')})
            history = RecoveryHistory(length=8, physical_hz=120, task_budget_steps=1200,
                                      recovery_budget_steps=120)
            windows, masks = [], []
            for index in range(max(needed) + 1):
                observation = read_tree(handle['observations'][str(index)])
                features = recovery_features(observation, tactile,
                                             recovery_remaining_steps=observation['remaining_recovery_steps'])
                window, mask = history.append(features)
                if index in needed:
                    windows.append(window.copy())
                    masks.append(mask.copy())
            with torch.inference_mode():
                encoded = model.encoder(np.stack(windows), np.stack(masks)).cpu().numpy()
            states = dict(zip(needed, encoded))
            for record in records:
                transition = Transition(states[record['observation_index']], record['action'], record['reward'],
                                        states[record['next_observation_index']], record['terminated'],
                                        record['truncated'], record['bootstrap_allowed'])
                rows.append(vars(transition))
                sources.append({'parent_scene_id': parent, 'action_index': record['action_index']})
            parents.append({'split': split, 'skill': skill_name, 'parent_scene_id': parent,
                            'episode_path': str(episode_path), 'version': metadata.get('versions', {}),
                            'terminal_reason': json.loads(handle.attrs['outcome'])['reason']})
            print(f'{parent}: {len(records)} transitions encoded', flush=True)
    if not rows:
        raise ValueError('no recovery transitions')
    return {**{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
            'metadata': {'schema': 'evotac.recovery_replay_dataset.v2', 'observation_dim': 128,
                         'action_dim': 7, 'split': split, 'skill_names': [skill_name], 'parents': parents,
                         'sources': sources, 'discount': 0.99,
                         'warmstart_sha256': hashlib.sha256(Path(warmstart).read_bytes()).hexdigest()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='+', type=Path)
    parser.add_argument('--warmstart', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('train', 'dev', 'test'), default='train')
    parser.add_argument('--skill-name', default='small_lift_adjust_reapproach')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = extract(args.episodes, args.warmstart, split=args.split, skill_name=args.skill_name,
                     device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{k: v for k, v in result.items() if k != 'metadata'},
                        metadata=json.dumps(result['metadata'], allow_nan=False))
    print(json.dumps({'status': 'written', 'transitions': len(result['reward']),
                      'parents': len(result['metadata']['parents']),
                      'positive_rewards': int((result['reward'] > 0).sum()), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
