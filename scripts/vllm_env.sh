#!/usr/bin/env bash
# vLLM 0.21.0+cu129 needs CUDA 12 runtime libs bundled in the venv.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/.venv/bin:${PATH}"
export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
