#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "A CUDA-enabled PyTorch installation and NVIDIA GPU are required"; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB VRAM")'
.venv/bin/chess-self doctor
.venv/bin/chess-self train --device cuda --data data/processed --checkpoint models/maia3-23m-ponder.pt --output "${1:-models/runs/heads-01}" --batch-size 16 --epochs 8
