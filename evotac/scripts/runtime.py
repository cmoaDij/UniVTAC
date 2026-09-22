"""Shared livestream startup. Configuration is read before simulator imports."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import signal
from datetime import datetime, timezone
from pathlib import Path

from evotac.config import ROOT, load_config, run_paths
from evotac.data.schemas import json_value
from evotac.envs.state_replay import digest
from evotac.perf.runtime_probe import gpu_snapshot


def parser_for(description, *, app_launcher=True):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1_insert_hole.yaml")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    parser.add_argument("--seed", type=int, default=0)
    # AppLauncher itself is safe before application startup; task imports are
    # not.  A config-only dry run can skip the import (and the EULA bootstrap).
    if app_launcher:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(livestream=2, enable_cameras=True)
    parser.add_argument("--performance-profile", action="store_true",
                        help="Run without livestream/interactive viewport for speed profiling; sensor rendering stays enabled.")
    return parser


def launch(args):
    config = load_config(args.config)
    flush_every = getattr(args, "flush_every", None)
    if flush_every is not None:
        if not getattr(args, "performance_profile", False):
            raise ValueError("--flush-every is only allowed with --performance-profile")
        if type(flush_every) is not int or not 1 <= flush_every <= 128:
            raise ValueError("--flush-every must be an integer in [1, 128]")
        config["logging"]["flush_every"] = flush_every
    if args.headless and not args.performance_profile:
        raise ValueError("--headless is reserved for --performance-profile; validation uses --livestream 2")
    if args.performance_profile:
        args.headless = True
        args.livestream = 0
    elif args.livestream not in (1, 2):
        raise ValueError("Use --livestream 2 (or 1) for rendering validation")
    paths = run_paths(config, args.run_id)
    paths["dataset_root"].mkdir(parents=True, exist_ok=False)
    paths["run_root"].mkdir(parents=True, exist_ok=False)
    source = ROOT.parent
    tracked = subprocess.check_output(["git", "ls-files", "envs", "third_party/TacEx", "task_config", "scripts"], cwd=source, text=True).splitlines()
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in tracked if (source/name).is_file()}
    own_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for directory in ("envs", "data", "evaluation", "learning", "perf", "scripts")
                  if (ROOT/directory).exists() for p in (ROOT/directory).rglob("*.py")}
    asset_files = [source/"assets/objects"/name for name in ("TestTube.usd", "TestTubeBase.usd", "TestTubeHoleSlot.usd")]
    asset_files += list((source/"assets/embodiments/franka").glob("*"))
    asset_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in asset_files if p.is_file()}
    versions = {"schema": config["schema_version"], "action": config["controller"]["action_version"],
                "prefix": config["replay"]["protocol_version"], "config_sha256": digest(config),
                "upstream_files_sha256": digest(hashes), "evotac_code_sha256": digest(own_hashes),
                "task_assets_sha256": digest(asset_hashes), "isaac_sim": "5.1", "isaac_lab": "2.3",
                "upstream_git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()}
    launch_mode = "headless_profile" if args.performance_profile else "livestream"
    manifest = {"data_kind": "control_rollout", "versions": versions, "config": config,
                "command": sys.argv, "launch_environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "device": args.device, "seed": args.seed, "backend": "taxim", "launch_mode": launch_mode,
                "gpu_snapshot_before": gpu_snapshot(),
                "upstream_file_hashes": hashes, "evotac_file_hashes": own_hashes,
                "task_asset_hashes": asset_hashes, "status": "starting"}
    (paths["dataset_root"] / "manifest.json").write_text(json.dumps(json_value(manifest), indent=2))
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    launcher = AppLauncher(args)
    manifest.update(status="application_started", rendering={
        "experience": launcher._sim_experience_file,
        "livestream": launcher._livestream, "headless": bool(args.headless),
        "performance_profile": bool(args.performance_profile), "enable_cameras": True,
        "gpu_snapshot_after_start": gpu_snapshot()})
    (paths["dataset_root"] / "manifest.json").write_text(json.dumps(json_value(manifest), indent=2))
    # Restore catchable interrupts after SimulationApp installs its immediate
    # shutdown handler, so rollout finally blocks can flush before Kit closes.
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    return config, paths, versions, launcher


def build_wrapper(config, run_id, versions, device):
    from evotac.envs.factory import create_task
    from evotac.envs.univtac_rl_wrapper import UniVTACRLWrapper
    return UniVTACRLWrapper(create_task(config, run_id, device), config, run_id, versions)
