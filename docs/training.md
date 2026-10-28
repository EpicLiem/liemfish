# GPU training

Use the [remote training guide](lambda-quickstart.md) to transfer a verified bundle and run a detached job.

## Machine and transfer

Use Linux with Python 3/venv support, an NVIDIA GPU and a working driver. Setup installs uv when needed and creates a Python 3.12 environment from the pinned lock. The transfer bundle includes upstream source so Git is needed only for code-only setup. Completed training runs used a single A100 40 GB. Allow disk space for PyTorch/CUDA dependencies plus multiple approximately 109 MB checkpoints and data snapshots.

Copy the repository to the GPU machine, **including `data/`**, excluding `.venv/`, caches and `reports/cpu-smoke*/`. Do not copy a Mac virtual environment to Linux. The source and pinned pretrained checkpoint can be restored by setup. Keep the original raw manifests and processed splits together.

Run from the project root:

```sh
nvidia-smi
bash scripts/setup.sh
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

If PyTorch reports no CUDA, fix the driver/build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before proceeding. Training requires a working CUDA device. Re-run tests after changing dependencies and record the resulting package versions.

## Run

```sh
bash scripts/train-gpu.sh models/runs/heads-01
```

Watch the first batch for GPU memory and throughput. If it runs out of memory, use a fresh output directory and lower `--batch-size` through `chess-self train`. Every run refuses an existing nonempty output directory; an interrupted run can be restarted as a new experiment. Optimizer-resume is not implemented.

The script runs the unchanged validation baseline, then trains heads with mixed precision. It saves the best complete checkpoint by validation move NLL, retaining the baseline if no epoch improves. Gradient clipping and loss scaling are enabled.

Review `run.json`: confirm successful optimizer steps, finite loss, improvement over the unchanged baseline, timing errors against the empirical timing baseline, and per-control results. Do not choose hyperparameters on test results. The sparse rapid/classical controls may require more data; only 748 standard games survived preparation, and a large part of the historical training set predates current play.

After selection, retain the chosen checkpoint and its `run.json`. Use the [timing experiment guide](timing-experiments.md) for evaluation and export commands.
