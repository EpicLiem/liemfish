"""Download only the pinned checkpoint and verify it before use."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models/maia3-23m-ponder.pt"
SHA256 = "0ac9fb74256f9be7d74808404364b90130c4d193f999c45b01e9b5ef64281bd1"
URL = "https://huggingface.co/UofTCSSLab/Maia3-23M-ponder/resolve/34ec2be9d651102e4ad0e9eda3d838acd69f5f9a/maia3-23m-ponder.pt"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    if DESTINATION.exists():
        if digest(DESTINATION) != SHA256:
            raise SystemExit("Existing checkpoint checksum mismatch; refusing to overwrite")
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        temporary = DESTINATION.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        if digest(temporary) != SHA256:
            raise SystemExit("Download checksum mismatch; incomplete file retained as .part")
        temporary.rename(DESTINATION)
    print(f"Verified {DESTINATION.name}: {SHA256}")
