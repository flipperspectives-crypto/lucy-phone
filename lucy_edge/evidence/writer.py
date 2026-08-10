"""Atomic writes, hashing and exception sanitization for evidence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

_SECRET_KEYS = (
    "token",
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "credential",
    "session",
)


def sanitize_exception(exc: BaseException) -> str:
    """Safe, credential-free exception summary."""
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SECRET_KEYS)


def sanitize_for_evidence(obj: Any, path: str = "") -> Any:
    """Recursively drop values whose keys look like secrets."""
    if isinstance(obj, dict):
        return {
            k: (sanitize_for_evidence(v, f"{path}.{k}") if not _looks_secret(k) else "<redacted>")
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [sanitize_for_evidence(item, f"{path}[]") for item in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_evidence(item, f"{path}[]") for item in obj]
    return obj


def atomic_write_text(path: str, text: str, fsync: bool = True) -> str:
    """Write ``text`` to ``path`` atomically.

    Writes to a temp file in the same directory, fsyncs it, atomically renames
    over the target, then fsyncs the directory where the platform supports it.
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp, str(target))
        if fsync:
            _fsync_dir(parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(target)


def _fsync_dir(directory: Path) -> None:
    """fsync a directory if the platform supports it (best effort)."""
    if os.name == "nt":
        return
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        dir_fd = os.open(str(directory), flags)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> Optional[str]:
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
