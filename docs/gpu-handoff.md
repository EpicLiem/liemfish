# GPU training handoff

The local preparation milestone is complete. Next is an actual measured training experiment on your data. The project does not open an SSH connection or provision cloud resources itself.

For the prepared transfer bundle and optional SSH deployment helper, use [Lambda quick start](lambda-quickstart.md). The helper connects only when explicitly invoked with the instance endpoint. The detached full pipeline runs setup, a real GPU preflight, and training in order.

## Machine and transfer

Use Linux with Python 3/venv support, an NVIDIA GPU and a working driver. Setup installs uv when needed and creates a Python 3.12 environment from the pinned lock. The transfer bundle includes upstream source so Git is needed only for code-only setup. A single A100 is the intended rental target. No VRAM or duration guarantee has been measured. Allow disk space for PyTorch/CUDA dependencies plus multiple approximately 109 MB checkpoints and data snapshots.

Copy the entire `chess-self` folder to the authorized GPU machine, **including `data/`**, excluding `.venv/`, caches and `reports/cpu-smoke*/`. Do not copy a Mac virtual environment to Linux. The source and pinned pretrained checkpoint can be restored by setup. Keep the original raw manifests and processed splits together.

Run from the project root:

```sh
nvidia-smi
bash scripts/setup.sh
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

If PyTorch reports no CUDA, fix the driver/build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before proceeding. Do not silently run the full experiment on CPU. Re-run tests after changing dependencies and record the resulting package versions.

## Run

```sh
bash scripts/train-gpu.sh models/runs/heads-01
```

Watch the first batch for GPU memory and throughput. If it runs out of memory, use a fresh output directory and lower `--batch-size` through `chess-self train`. Every run refuses an existing nonempty output directory; an interrupted run can be restarted as a new experiment. Optimizer-resume is not implemented.

The script runs the unchanged validation baseline, then trains heads with mixed precision. It saves the best complete checkpoint by validation move NLL, retaining the baseline if no epoch improves. Gradient clipping and loss scaling are enabled. CUDA execution remains to be verified on the supplied machine; CPU smoke execution does not validate GPU operators or available VRAM.

Review `run.json`: confirm successful optimizer steps, finite loss, improvement over the unchanged baseline, timing errors against the empirical timing baseline, and per-control results. Do not choose hyperparameters on test results. The sparse rapid/classical controls may require more data; only 748 standard games survived preparation, and a large part of the historical training set predates current play.

After selection, use the README commands for the untouched test comparison and ONNX export. Keep all generated weights local until you decide otherwise. Build and benchmark the browser application after model selection.

## Evidence already on this Mac

- `reports/accounts.json`: signed-in account provenance and archive counts.
- `data/raw/*.manifest.json`: URLs, dates and hashes for all downloads.
- `data/processed/audit.json`: exclusions, clock coverage, repeated positions and game split IDs.
- `reports/cpu-smoke-final/run.json`: real pretrained forward/backward and checkpoint selection on four training and four validation positions; two optimizer updates only.
- `reports/readiness.json`: strict load and legal inference; no CUDA GPU available here.
- `reports/baseline-validation.json`: unchanged checkpoint metrics on all 1,844 validation positions: 50.76% top-1, 79.45% top-3 agreement; 5.47 seconds timing MAE versus 6.26 seconds for the training-derived empirical baseline. Test performance remains unmeasured.
- `reports/export-pretrained.json`: nine PyTorch/ONNX CPU parity cases, not a browser benchmark.

No personalized model is claimed until a real run improves held-out imitation metrics.
