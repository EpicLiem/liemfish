"""Public PGN archive snapshots, with verified, non-destructive resume."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import chess.pgn
import requests

USER_AGENT = "chess-self/0.1 (personal public game archive downloader)"
TIMEOUT = (15, 120)
MAX_ATTEMPTS = 4


def _now():
    return datetime.now(timezone.utc).isoformat()


def _request(session, url, *, params=None, accept="application/x-chess-pgn"):
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(0.25)
        response = None
        try:
            response = session.get(url, params=params, timeout=TIMEOUT,
                                   headers={"Accept": accept}, allow_redirects=False)
            if response.status_code in (429, 500, 502, 503, 504):
                response.raise_for_status()
            if response.status_code != 200:
                raise requests.HTTPError(f"HTTP {response.status_code}: {url}", response=response)
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            retryable = response is None or response.status_code in (429, 500, 502, 503, 504)
            if not retryable or attempt == MAX_ATTEMPTS - 1:
                raise
            delay = 60 if response is not None and response.status_code == 429 else 2 ** attempt
            retry_after = response.headers.get("Retry-After") if response is not None else None
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError):
                        pass
            if delay > 300:
                raise requests.HTTPError("Retry-After exceeds five-minute retry budget", response=response) from exc
            if response is not None:
                response.close()
            time.sleep(delay)


def _atomic(path, content, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(name, path)
        else:
            # Linking fails if another process or a previous run already owns the path.
            os.link(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _pgn_stats(content):
    text = content.decode("utf-8-sig")
    if text.strip() and not text.lstrip().startswith('[Event "'):
        raise ValueError("Response is not a PGN export")
    stream = io.StringIO(text)
    count = clocks = evals = 0
    dates = []
    while (game := chess.pgn.read_game(stream)) is not None:
        if game.errors or game.headers.get("White", "?") == "?" or game.headers.get("Black", "?") == "?":
            raise ValueError("Invalid PGN game or missing player headers")
        count += 1
        comments = [node.comment for node in game.mainline()]
        clocks += any("[%clk " in comment for comment in comments)
        evals += any("[%eval " in comment for comment in comments)
        date = game.headers.get("UTCDate", game.headers.get("Date", ""))
        if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", date):
            dates.append(date)
    if text.strip() and count == 0:
        raise ValueError("Nonempty export contains no games")
    return {"games": count, "games_with_clocks": clocks, "games_with_evals": evals,
            "first_game_date": min(dates, default=None), "last_game_date": max(dates, default=None)}


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def fetch_games(platform: str, username: str, output: Path) -> dict:
    """Fetch all publicly archived finished games into an output directory.

    Existing exports are hash-checked and reused, never refreshed implicitly.
    Use a fresh output directory for an updated snapshot of mutable archives.
    The returned manifest reports partial failures; invalid local state raises.
    """
    platform = platform.lower()
    if platform not in {"lichess", "chess.com"}:
        raise ValueError("platform must be 'lichess' or 'chess.com'")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", username):
        raise ValueError("Invalid username")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{platform}-{username.lower()}"
    manifest_path = output / f"{prefix}.manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest cannot be a symlink")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "platform": platform, "username": username,
        "created_at": _now(), "files": [],
        "resume_policy": "Existing exports are immutable snapshots; use a new output directory to refresh.",
    }
    if manifest.get("platform") != platform or manifest.get("username", "").lower() != username.lower():
        raise ValueError("Manifest account mismatch")
    existing = {}
    for record in manifest["files"]:
        filename = record["file"]
        if Path(filename).name != filename or not filename.startswith(prefix + "-"):
            raise ValueError("Unsafe manifest file path")
        path = output / filename
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Missing or altered raw export: {filename}")
        existing[filename] = record
    manifest.update(last_attempt_at=_now(), failures=[], complete=False)

    def save():
        manifest["total_games"] = sum(r.get("games", 0) for r in manifest["files"])
        _atomic(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode(), replace=True)

    def download(session, url, filename, params=None, is_index=False):
        path = output / filename
        if filename in existing:
            return path.read_bytes()
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Untracked raw export; refusing to overwrite: {path}")
        with _request(session, url, params=params, accept="application/json" if is_index else "application/x-chess-pgn") as response:
            content = response.content
            if is_index:
                _archive_urls(content, username)
                stats = {}
            else:
                content_type = response.headers.get("Content-Type", "").lower()
                if "html" in content_type or "json" in content_type:
                    raise ValueError(f"Unexpected PGN Content-Type: {content_type}")
                stats = _pgn_stats(content)
            record = {"file": filename, "url": response.url, "params": params or {},
                      "fetched_at": _now(), "sha256": _sha(content), "bytes": len(content),
                      "http_date": response.headers.get("Date"),
                      "last_modified": response.headers.get("Last-Modified"),
                      "etag": response.headers.get("ETag"), **stats}
        _atomic(path, content)
        manifest["files"].append(record)
        existing[filename] = record
        save()
        return content

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        # Public APIs need neither ambient netrc credentials nor browser cookies.
        session.trust_env = False
        if platform == "lichess":
            targets = [(f"https://lichess.org/api/games/user/{username}", f"{prefix}-all.pgn",
                        {"clocks": "true", "evals": "false", "opening": "false", "sort": "dateAsc"})]
        else:
            url = f"https://api.chess.com/pub/player/{username.lower()}/games/archives"
            try:
                content = download(session, url, f"{prefix}-archives.json", is_index=True)
                targets = [(url + "/pgn", f"{prefix}-{'-'.join(url.split('/')[-2:])}.pgn", None)
                           for url in _archive_urls(content, username)]
            except (requests.RequestException, ValueError) as exc:
                manifest["failures"].append({"url": url, "error": str(exc)})
                save()
                return manifest
        for url, filename, params in targets:
            try:
                download(session, url, filename, params)
            except (requests.RequestException, ValueError) as exc:
                manifest["failures"].append({"url": url, "error": str(exc)})
                save()
        manifest["complete"] = not manifest["failures"]
        manifest["completed_at"] = _now() if manifest["complete"] else None
        save()
    return manifest


def _archive_urls(content, username):
    data = json.loads(content)
    urls = data.get("archives") if isinstance(data, dict) else None
    if not isinstance(urls, list):
        raise ValueError("Invalid archives index")
    pattern = rf"https://api\.chess\.com/pub/player/{re.escape(username.lower())}/games/\d{{4}}/(0[1-9]|1[0-2])"
    if any(not isinstance(url, str) or not re.fullmatch(pattern, url) for url in urls):
        raise ValueError("Unexpected archive URL")
    return sorted(set(urls))
