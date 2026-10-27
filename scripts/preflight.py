"""Verify real forward/backward execution before launching a paid GPU run."""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch
from chess_self.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("GPU preflight failed: CUDA is unavailable. No full training started.")
    if args.output.exists():
        raise SystemExit("Preflight output already exists; choose a fresh path")
    info = {"device": args.device, "gpu_verified": False, "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    if args.device == "cuda":
        info.update(gpu=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
        info["nvidia_smi"] = result.stdout
    start = time.perf_counter()
    report = train(Path("data/processed"), Path("models/maia3-23m-ponder.pt"), args.output,
                   device=args.device, smoke=True)
    info.update(smoke_seconds=time.perf_counter() - start, optimizer_steps=report["optimizer_steps"],
                skipped_updates=report["skipped_amp_updates"])
    if report["optimizer_steps"] == 0:
        raise SystemExit("Preflight produced no successful optimizer updates; full training blocked")
    if args.device == "cuda":
        torch.cuda.synchronize()
        info.update(gpu_verified=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    info["note"] = "Two-batch smoke test. Full batch-size memory and throughput are measured during the actual run."
    (args.output / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
