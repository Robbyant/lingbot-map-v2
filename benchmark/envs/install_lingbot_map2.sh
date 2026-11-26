#!/usr/bin/env bash
# Copyright 2026 The Robbyant Team Authors. All rights reserved.
# Create one CUDA environment for inference and all supported benchmarks.
# Usage: bash benchmark/envs/install_lingbot_map2.sh [--append]
# Existing environments are never removed. --append uses lingbot-map2 as-is.
set -euo pipefail

append=false
case "${1:-}" in
    "") ;;
    --append) append=true ;;
    -h|--help)
        echo "Usage: bash benchmark/envs/install_lingbot_map2.sh [--append]"
        echo "Creates lingbot-map2 with Python 3.11 and PyTorch 2.8 / CUDA 12.8."
        echo "--append: install package/benchmark dependencies in an existing environment."
        exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Expected at most one option." >&2
    exit 2
fi
command -v conda >/dev/null || { echo "Install Conda first; see README.md." >&2; exit 1; }
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "$repo_dir/pyproject.toml" ]]; then
    echo "Run this script from a source checkout." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx 'lingbot-map2'; then
    if [[ "$append" != true ]]; then
        echo "Environment lingbot-map2 exists. Use --append to keep it and install dependencies." >&2
        exit 1
    fi
else
    if [[ "$append" == true ]]; then
        echo "Environment lingbot-map2 does not exist; omit --append to create it." >&2
        exit 1
    fi
    conda create -n lingbot-map2 python=3.11 -y
    conda run -n lingbot-map2 python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi

# Keep the user's existing CUDA build on append; diagnose incompatible builds
# before pip can replace them. Dependency declarations live in pyproject.toml.
conda run -n lingbot-map2 python -c '
import torch
version = tuple(map(int, torch.__version__.split("+")[0].split(".")[:2]))
if version < (2, 8) or torch.version.cuda is None:
    raise SystemExit("Install CUDA-enabled PyTorch >= 2.8 first; see README.md.")
'
conda run -n lingbot-map2 python -m pip install -e "$repo_dir[benchmark]"
echo "Ready. Run: conda activate lingbot-map2"
echo "Then configure dataset paths and your causal checkpoint in benchmark/configs/."
