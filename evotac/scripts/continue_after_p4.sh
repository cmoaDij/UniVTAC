#!/usr/bin/env bash
set -euo pipefail
cd /data/ZED/UniVTAC
exec /data/ZED/conda/envs/UniVTAC/bin/python -u -m evotac.scripts.continue_validation "$@"
