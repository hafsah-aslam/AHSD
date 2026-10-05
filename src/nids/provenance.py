"""Hashing and run provenance (git hash, data SHA-256, RQ hash)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RQ_FILE = ROOT / "RESEARCH_QUESTIONS.md"


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _head() -> tuple[str, bool]:
    """(HEAD commit, tracked-files-dirty) of the repository right now."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out, bool(dirty)
    except Exception:
        return "unknown", True


# Captured once, when this module is first imported (i.e. at process start, before any
# result is written). This is the code the process runs, whatever HEAD becomes later.
_CODE_HEAD, _CODE_DIRTY = _head()
CODE_COMMIT = {"code_commit": _CODE_HEAD, "code_dirty": _CODE_DIRTY,
               "code_commit_captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}


def require_clean_tree(what: str = "this phase") -> None:
    """Refuse to start work from a tree with uncommitted changes to tracked files."""
    if CODE_COMMIT["code_dirty"] or CODE_COMMIT["code_commit"] == "unknown":
        raise SystemExit(f"refusing to start {what}: working tree has uncommitted changes to tracked files "
                         f"(code_commit {CODE_COMMIT['code_commit']}). Commit first.")


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def rq_sha256() -> str:
    return sha256_file(RQ_FILE) if RQ_FILE.exists() else "missing"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def provenance(config: dict | None = None, data_sha256: dict | None = None) -> dict:
    return {
        **CODE_COMMIT,                 # compared by provenance checks
        "results_head": git_hash(),    # HEAD at write time; informational only
        "rq_sha256": rq_sha256(),
        "data_sha256": data_sha256 or {},
        "config": config or {},
        "timestamp_utc": now_iso(),
        "python": platform.python_version(),
        "host": platform.node(),
    }


def write_json(path: str | os.PathLike, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default, sort_keys=False)
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
