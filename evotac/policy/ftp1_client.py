"""Local subprocess transport keeps FTP-1 dependencies out of Isaac's Python."""
import json
import hashlib
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np

from evotac.config import ROOT, output_path


class FTP1Client:
    def __init__(self, config, run_directory):
        self.config = config
        self.directory = output_path(Path(run_directory)/"policy")
        self.directory.mkdir(parents=True, exist_ok=True)
        source = (ROOT/config["source_root"]).resolve()
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != config["source_revision"]:
            raise ValueError("FTP-1 source revision does not match the pinned configuration")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config["model_gpu"]), JAX_PLATFORMS="cpu",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false", USE_TF="0", TOKENIZERS_PARALLELISM="false",
                           OPENPI_DATA_HOME=str(ROOT/".cache/openpi"), HF_HOME=str(ROOT/".cache/huggingface"),
                           TORCH_HOME=str(ROOT/".cache/torch"), XDG_CACHE_HOME=str(ROOT/".cache"),
                           PYTHONPATH=os.pathsep.join(map(str, [ROOT.parent, source/"src", source, source/"packages/openpi-client/src"])))
        self.log = (self.directory/"worker.log").open("w")
        self.process = subprocess.Popen([
            str(ROOT/config["python"]), "-u", "-m", "evotac.scripts.ftp1_worker",
            "--checkpoint", str(ROOT/config["checkpoint"]), "--domain", config["domain"],
            "--steps", str(config["num_inference_steps"]), "--seed", str(config.get("inference_seed", 0))], cwd=ROOT.parent, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.request_id = 0
        try:
            self.metadata = self._receive(config["startup_timeout_seconds"])
            if self.metadata["status"] != "ready":
                raise RuntimeError(f"FTP-1 worker startup failed: {self.metadata}")
            self.metadata.update(source_revision=revision, config=config)
            provenance = ROOT/"checkpoints/ftp1_univtac/verified_source.json"
            self.metadata["checkpoint_provenance"] = json.loads(provenance.read_text())
            self.metadata["adapter_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [ROOT/"policy/ftp1_adapter.py", ROOT/"policy/ftp1_client.py", ROOT/"scripts/ftp1_worker.py"]}
            (self.directory/"manifest.json").write_text(json.dumps(self.metadata, indent=2))
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.process.stdout], [], [], max(0, deadline-time.monotonic()))
            if not ready:
                break
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"FTP-1 worker exited; inspect {self.directory/'worker.log'}")
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                self.log.write("worker stdout: "+line)
                self.log.flush()
                continue
            if result.get("status") == "error":
                raise RuntimeError(result["error"])
            return result
        raise TimeoutError(f"FTP-1 worker timed out; inspect {self.directory/'worker.log'}")

    def infer(self, inputs, prompt):
        index = self.request_id
        self.request_id += 1
        input_path = self.directory/f"input_{index:05d}.npz"
        output_path = self.directory/f"chunk_{index:05d}.npy"
        np.savez_compressed(input_path, **inputs)
        request = {"request_id": index, "input_path": str(input_path), "output_path": str(output_path), "prompt": str(prompt)}
        self.process.stdin.write(json.dumps(request)+"\n")
        self.process.stdin.flush()
        result = self._receive(self.config["inference_timeout_seconds"])
        if result.get("status") != "inferred" or result.get("request_id") != index:
            raise RuntimeError("Mismatched inference response")
        with (self.directory/"inferences.jsonl").open("a") as f:
            f.write(json.dumps({**request, **result})+"\n")
        return np.load(output_path, allow_pickle=False), result

    def reset_seed(self, seed):
        """Start a declared continuation RNG stream without reloading weights."""
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("inference seed must be an integer in [0, 2**32)")
        index = self.request_id
        self.request_id += 1
        request = {"operation": "reset_seed", "request_id": index, "seed": seed}
        self.process.stdin.write(json.dumps(request)+"\n")
        self.process.stdin.flush()
        result = self._receive(self.config["inference_timeout_seconds"])
        if (result.get("status") != "seed_reset" or result.get("seed") != seed
                or result.get("request_id") != index):
            raise RuntimeError("Mismatched inference seed reset response")
        with (self.directory/"rng_resets.jsonl").open("a") as handle:
            handle.write(json.dumps({**request, **result})+"\n")
        return result

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
