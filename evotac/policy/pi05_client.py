"""Isolated subprocess transport for the optional LeRobot π0.5 runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np

from evotac.config import ROOT, output_path


class PI05Client:
    def __init__(self, config, run_directory):
        self.config = config
        if config.get("use_tactile") is not False:
            raise ValueError("π0.5 baseline must declare use_tactile=false")
        self.directory = output_path(Path(run_directory) / "policy")
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = (ROOT / config["checkpoint"]).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"π0.5 checkpoint is unavailable: {checkpoint}")
        environment = dict(os.environ,
                           CUDA_VISIBLE_DEVICES=str(config["model_gpu"]),
                           TOKENIZERS_PARALLELISM="false",
                           PYTHONPATH=str(ROOT.parent))
        self.log = (self.directory / "worker.log").open("w")
        self.process = subprocess.Popen(
            [str(ROOT / config["python"]), "-u", "-m", "evotac.scripts.pi05_worker",
             "--checkpoint", str(checkpoint), "--device", "cuda:0",
             "--steps", str(config["num_inference_steps"]),
             *( ["--stats", str((ROOT / config["stats"]).resolve())]
                if config.get("stats") else [] )],
            cwd=ROOT.parent, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.request_id = 0
        try:
            self.metadata = self._receive(config["startup_timeout_seconds"])
            if self.metadata.get("status") != "ready":
                raise RuntimeError(f"π0.5 worker startup failed: {self.metadata}")
            self.metadata["config"] = config
            (self.directory / "manifest.json").write_text(json.dumps(self.metadata, indent=2))
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.process.stdout], [], [], deadline - time.monotonic())
            if not ready:
                break
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"π0.5 worker exited; inspect {self.directory/'worker.log'}")
            result = json.loads(line)
            if result.get("status") == "error":
                raise RuntimeError(result["error"])
            return result
        raise TimeoutError(f"π0.5 worker timed out; inspect {self.directory/'worker.log'}")

    def infer(self, inputs, prompt):
        index = self.request_id
        self.request_id += 1
        input_path = self.directory / f"input_{index:05d}.npz"
        output_path = self.directory / f"chunk_{index:05d}.npy"
        np.savez_compressed(input_path, **inputs)
        self.process.stdin.write(json.dumps({"operation": "infer", "request_id": index,
                                             "input_path": str(input_path),
                                             "output_path": str(output_path),
                                             "prompt": str(prompt)}) + "\n")
        self.process.stdin.flush()
        result = self._receive(self.config["inference_timeout_seconds"])
        if result.get("status") != "inferred" or result.get("request_id") != index:
            raise RuntimeError("mismatched π0.5 inference response")
        with (self.directory / "inferences.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        return np.load(output_path, allow_pickle=False), result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"operation":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.log.close()
