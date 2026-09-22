"""Load real FTP-1 weights and infer on a recorded deployment observation."""
import argparse
import json
from pathlib import Path
import traceback

import h5py
import yaml

from evotac.config import ROOT, output_path
from evotac.data.rollout_logger import read_tree
from evotac.policy.ftp1_adapter import encode_observation, decode_chunk
from evotac.policy.ftp1_client import FTP1Client


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--policy-config", type=Path, default=ROOT/"configs/ftp1_insert_hole.yaml")
    args = p.parse_args()
    directory = output_path(Path("runs/ftp1")/args.run_id)
    directory.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.policy_config.read_text())
    checks, client = {"status": "starting", "episode": str(args.episode)}, None
    try:
        with h5py.File(args.episode, "r") as handle:
            observation = read_tree(handle["observations/0"])
        inputs, base = encode_observation(observation)
        client = FTP1Client(config, directory)
        chunk, result = client.infer(inputs, config["prompt"])
        actions = decode_chunk(chunk, base, client.metadata["action_joint_rep"])
        checks.update(status="passed", model=client.metadata, inference=result,
                      action_shape=list(chunk.shape), first_joint_target=list(actions[0].values))
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        (directory/"checks.json").write_text(json.dumps(checks, indent=2))
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
