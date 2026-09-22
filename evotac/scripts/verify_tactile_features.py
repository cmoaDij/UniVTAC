"""Strict published checkpoint loading, old/new frames, frozen BN and feature isolation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from evotac.config import ROOT, output_path
from evotac.data.legacy_demonstrations import LegacyDemonstrations
from evotac.data.rollout_logger import read_tree
from evotac.envs.state_replay import digest
from evotac.policy.tactile_features import FrozenTactileEncoder, recovery_features, SENSORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="ftp1_livestream_full_01")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    directory = output_path(Path("runs/phase2")/args.run_id)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    provenance = json.loads((ROOT/"checkpoints/univtac_release/encoder_source.json").read_text())
    encoder = FrozenTactileEncoder(ROOT/"checkpoints/univtac_release/encoder.pth", provenance["sha256"], args.device)
    before = {k: v.clone() for k, v in encoder.named_buffers()}
    checks = json.loads((ROOT/"runs/phase1"/args.source_run/"checks.json").read_text())
    features, raw = [], []
    with h5py.File(checks["episode_path"], "r") as handle:
        indices = np.linspace(0, len(handle["observations"])-1, 8, dtype=int)
        for index in indices:
            observation = read_tree(handle["observations"][str(index)])
            feature = recovery_features(observation, encoder, recovery_remaining_steps=120)
            contaminated = copy.deepcopy(observation)
            contaminated.update(actor={"secret": 99}, task_success=True, condition="privileged")
            assert digest(feature) == digest(recovery_features(contaminated, encoder, recovery_remaining_steps=120))
            features.append(feature["tactile_embeddings"])
            raw.extend(observation["images"]["tactile"][s]["rgb_marker"]["data"] for s in SENSORS)
    legacy = LegacyDemonstrations()
    rows = [r for r in legacy.manifest() if r["split"] == "dev"][:3]
    old = []
    for row in rows:
        obs = legacy.read_observation(row, 0)
        old.extend(obs["tactile"][s]["rgb_marker"] for s in SENSORS)
    old_embeddings = encoder(np.stack(old))
    probe = np.stack(raw[:2])
    value = encoder(probe)
    encoder.train(True)
    repeated = encoder(probe)
    assert np.array_equal(value, repeated)
    assert all(torch.equal(before[k], v) for k, v in encoder.named_buffers())
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    altered = encoder(255-probe)
    sensitivity = float(np.max(np.abs(altered-value)))
    assert sensitivity > 1e-6
    new = np.stack(features)
    temporal_difference = float(np.max(np.abs(new[:, 0]-new[:1, 0])))
    assert temporal_difference > 1e-6
    np.savez_compressed(directory/"features.npz", new=new, legacy=old_embeddings, physics_observation_indices=indices)
    report = {"status": "passed", "provenance": provenance, "encoder": encoder.version,
              "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT/"policy/tactile_features.py", ROOT.parent/"encoder/network.py", ROOT.parent/"encoder/dataloader.py")},
              "new_feature_shape": list(new.shape), "legacy_feature_shape": list(old_embeddings.shape),
              "new_current_feature_std": float(new[:, 0].std()), "legacy_feature_std": float(old_embeddings.std()),
              "new_current_l2_mean": float(np.linalg.norm(new[:, 0], axis=-1).mean()),
              "new_current_across_time_max_difference": temporal_difference,
              "legacy_l2_mean": float(np.linalg.norm(old_embeddings, axis=-1).mean()),
              "repeat_max_error": float(np.max(np.abs(repeated-value))), "inverted_input_max_difference": sensitivity,
              "frozen_parameters": True, "batchnorm_buffers_unchanged": True, "privileged_input_isolated": True,
              "new_source_episode": checks["episode_path"], "legacy_sources": rows,
              "limitations": ["Forward compatibility and sensitivity do not prove recovery utility",
                              "Encoder pretraining used H=320,W=240; published ACT deploy instead resizes to 256x256. This feature interface explicitly uses the encoder training transform.",
                              "Legacy images preserve original codec round trip; historical training channel provenance remains unconfirmed",
                              "Contact pretraining dataset has not been fully content-audited"]}
    (directory/"checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
