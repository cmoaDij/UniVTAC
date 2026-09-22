"""Prepare FTP-1 in an isolated venv; run with the existing UniVTAC Python."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from evotac.config import ROOT, output_path


def main():
    config = yaml.safe_load((ROOT/"configs/ftp1_insert_hole.yaml").read_text())
    source = output_path(config["source_root"])
    # venv's interpreter is normally a symlink to the read-only base Python.
    # Validate the destination directory, without resolving that executable.
    venv = output_path(Path(config["python"]).parent.parent)
    python = venv/"bin/python"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/michaelyuancb/ftp1-policy.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", config["source_revision"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != config["source_revision"]:
        raise RuntimeError("Existing source checkout differs from the configured pin")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    import torch
    import torchvision
    constraints = output_path(".cache/ftp1_torch_constraints.txt")
    constraints.write_text(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n")
    environment = dict(os.environ, PIP_CACHE_DIR=str(ROOT/".cache/pip"), JAX_PLATFORMS="cpu", USE_TF="0")
    subprocess.run([str(python), "-m", "pip", "install", "-c", str(constraints), "-r", str(ROOT/"requirements_ftp1.txt")], env=environment, check=True)
    target = Path(subprocess.check_output([str(python), "-c", "import transformers; print(transformers.__path__[0])"], env=environment, text=True).strip()).resolve()
    if not target.is_relative_to(venv.resolve()):
        raise RuntimeError("Refusing to patch transformers outside the isolated venv")
    patches = source/"src/openpi/models_pytorch/transformers_replace"
    copied = []
    for patch in patches.rglob("*.py"):
        destination = target/patch.relative_to(patches)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(patch, destination)
        copied.append(str(patch.relative_to(patches)))
    environment["PYTHONPATH"] = os.pathsep.join(map(str, [source/"src", source, source/"packages/openpi-client/src"]))
    subprocess.run([str(python), "-c", "from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper; print('FTP1 import passed')"], env=environment, check=True)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--local"], env=environment, text=True)
    (venv/"requirements-installed.txt").write_text(frozen)
    (venv/"setup_manifest.json").write_text(json.dumps({"source_revision": revision, "base_python": sys.executable,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "transformer_patches": copied}, indent=2))
    print(f"Ready: {python}. Download weights with python -m evotac.scripts.download_ftp1_checkpoint")


if __name__ == "__main__":
    main()
