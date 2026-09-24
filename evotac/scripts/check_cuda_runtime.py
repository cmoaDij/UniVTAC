"""Check CUDA runtime context creation without allocating a model."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = {"device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        result.update(torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                      is_available=bool(torch.cuda.is_available()),
                      device_count=int(torch.cuda.device_count()))
        torch.empty(1, device=args.device)
        result.update(ok=True, reason="cuda_context_created")
    except BaseException as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
