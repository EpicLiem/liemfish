"""Upload and launch only when supplied an authorized instance's SSH endpoint."""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate(host, deployment, job):
    if not re.fullmatch(r"[A-Za-z0-9_-]+@[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("Use user@hostname or user@IPv4 as shown by Lambda")
    for value in (deployment, job):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
            raise ValueError("Deployment and job names must use letters, digits, hyphens or underscores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["deploy", "start", "status", "collect"])
    parser.add_argument("--host", required=True, help="For example ubuntu@INSTANCE_IP; no passwords")
    parser.add_argument("--identity", type=Path, help="Existing local SSH private-key path; never copied")
    parser.add_argument("--deployment", default="chess-self-gpu")
    parser.add_argument("--job", default="heads-01")
    parser.add_argument("--train", action="store_true", help="With deploy: run full pipeline after setup")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate(args.host, args.deployment, args.job)
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if args.identity:
        identity = args.identity.expanduser().resolve(strict=True)
        options += ["-i", str(identity)]
    remote_root = args.deployment + "/chess-self"

    def run(command):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    def ssh(command):
        run(["ssh", *options, args.host, command])

    def remote(command):
        ssh("cd " + shlex.quote(remote_root) + " && " + shlex.join(command))

    if args.action == "deploy":
        bundle = ROOT / "bundles/chess-self-gpu.tar.gz"
        if not bundle.is_file():
            raise SystemExit("Build the bundle first: python3 scripts/bundle.py")
        # A fresh directory makes duplicate deploys fail without overwriting work.
        ssh("mkdir -- " + shlex.quote(args.deployment))
        run(["scp", *options, str(bundle), args.host + ":" + args.deployment + "/bundle.tar.gz"])
        ssh("tar -xzf " + shlex.quote(args.deployment + "/bundle.tar.gz") + " -C " + shlex.quote(args.deployment))
        remote(["python3", "scripts/bundle.py", "--verify", "."])
        name = args.job if args.train else "setup-" + args.job
        remote(["python3", "scripts/jobs.py", "start", name, "--mode", "full" if args.train else "setup"])
    elif args.action == "start":
        remote(["python3", "scripts/jobs.py", "start", args.job, "--mode", "train"])
    elif args.action == "status":
        remote(["python3", "scripts/jobs.py", "status", args.job])
        ssh("tail -n 40 " + shlex.quote(remote_root + "/jobs/" + args.job + "/output.log"))
    else:
        destination = ROOT / "reports/collected" / args.deployment
        if not args.dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        failures = []
        for relative in ("jobs", "reports", "models/runs"):
            remote_path = remote_root + "/" + relative
            check = ["ssh", *options, args.host, "test -d " + shlex.quote(remote_path)]
            if args.dry_run:
                print(shlex.join(check))
            else:
                exists = subprocess.run(check, check=False)
                if exists.returncode == 1:
                    print(f"No {relative} directory yet; skipping", flush=True)
                    continue
                if exists.returncode:
                    failures.append(f"Could not check {relative}: SSH exit {exists.returncode}")
                    continue
            local = destination / relative.replace("/", "-")
            if not args.dry_run:
                local.mkdir(exist_ok=True)
            try:
                run(["scp", *options, "-r", args.host + ":" + remote_path + "/.", str(local)])
            except subprocess.CalledProcessError as error:
                failures.append(f"Could not collect {relative}: SCP exit {error.returncode}")
        if failures:
            raise SystemExit("Some transfers failed; other outputs were still attempted:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
