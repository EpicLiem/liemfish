# Remote training on Lambda

Use a single A100 instance with Lambda's normal Ubuntu image. This project installs its own isolated Python 3.12 environment and a pinned CUDA 12.6 PyTorch build. The image must have a working NVIDIA driver, `python3`, venv support and SSH access. GPU compatibility and throughput are checked on the actual machine.

## Deploy a training job

After the instance's SSH host key has been verified and the selected local key can connect:

```sh
cd liemfish
python3 scripts/remote.py deploy --host ubuntu@INSTANCE_IP --identity /absolute/path/to/existing-key --train
```

Replace only the endpoint and local key path. If your SSH agent/config already selects the key, omit `--identity`. The script uses normal SSH host-key checking; it does not weaken verification or install keys. Use `--dry-run` to print commands without connecting.

This creates `~/chess-self-gpu/chess-self` on the instance and:

1. Uploads the local bundle and verifies every included file.
2. Starts a detached job named `heads-01`.
3. Creates the environment, runs tests and checks the real dataset/checkpoint.
4. Runs a two-batch GPU forward/backward smoke test and checks successful updates.
5. Starts the initial heads-only training experiment, keeping the test set untouched.

Any failed step stops the job before subsequent steps execute. The script refuses an existing deployment directory or job name to protect prior work. Use `--deployment chess-self-second --job heads-02` for a separate run. A failure partway through upload leaves its directory in place; inspect it before retrying with a new deployment name.

For setup only, omit `--train`. That creates job `setup-heads-01`. Once it completes, start training with:

```sh
python3 scripts/remote.py start --host ubuntu@INSTANCE_IP --identity /absolute/path/to/existing-key
```

## Status and results

```sh
python3 scripts/remote.py status --host ubuntu@INSTANCE_IP --identity /absolute/path/to/existing-key
python3 scripts/remote.py collect --host ubuntu@INSTANCE_IP --identity /absolute/path/to/existing-key
```

Pass the same `--deployment` and `--job` if changed. For setup-only status, use `--job setup-heads-01`. Status prints the job state and last 40 log lines. Collection downloads saved runs, reports and job logs under `reports/collected/DEPLOYMENT/` locally. Collect after completion and confirm the files before terminating the instance. The scripts do not terminate the instance or stop rental billing.

The main outputs are `models/runs/heads-01/best.pt` and `run.json`. GPU preflight evidence is in `reports/gpu-preflight-heads-01/`; job output is `jobs/heads-01/output.log`. The baseline remains the winner unless personalization improves validation move NLL. A run that selects epoch zero is not a successful personalization result.

## Directly on the instance

If the bundle is transferred manually, extract into a fresh directory:

```sh
tar -xzf chess-self-gpu.tar.gz
cd chess-self
python3 scripts/bundle.py --verify .
python3 scripts/jobs.py start heads-01 --mode full
python3 scripts/jobs.py status heads-01
tail -f jobs/heads-01/output.log
```

`setup.sh` can verify again after runtime files exist using the manifest's runtime-aware check. It still verifies all source, raw-game, prepared-data and pretrained-weight hashes. Rebuild the bundle after editing code or the data; do not edit a transferred snapshot in place and assume its checksum verification will pass.

## Packaging and validation

```sh
python3 scripts/bundle.py
.venv/bin/python -m pytest -q
```

The bundle contains approximately 109 MB of pretrained weights plus source and games before compression. It excludes the local virtual environment, SSH credentials, caches, previous training runs and ONNX exports. Dependencies are fetched from the pinned lock on the instance. Allow setup time for the CUDA packages; this is separate from training throughput.


References: [Lambda connections](https://docs.lambda.ai/public-cloud/on-demand/connecting-to-an-instance/), [official PyTorch CUDA builds](https://pytorch.org/get-started/previous-versions/).
