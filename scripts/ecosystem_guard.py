#!/usr/bin/env python3
"""Ecosystem guard: enforce the strictly-offline rule on offline-by-design paths.

Scans every production source path that MUST NEVER make outbound network
calls or pull in external / cloud model libraries.  Fails (exit 1) if any
Python file there:

  * imports a network / cloud-capable module (urllib.request, socket, requests,
    httpx, grpc, openai, anthropic, huggingface, transformers, torch, ...),
  * carries an external-agent / export marker comment, or
  * contains a non-loopback URL literal (anything not 127.0.0.1 / localhost).

This is the technical backstop for Lucy's fail-closed / no-public-cloud /
nothing-imported-from-outside directive.  It is wired into the test suite
(see tests/test_ecosystem_guard.py) so a regression fails CI immediately.

NOTE: local plumbing libraries that are installed on-device and never dial out
(aiohttp + websockets for the loopback server, pydantic for config, numpy for
the brain, PyYAML, aiosqlite) are intentionally NOT banned here -- they are
being removed in a later from-scratch pass (Phase 2).  Once gone, extend
NETWORK_MODULES to ban them too.

Usage:  python3 scripts/ecosystem_guard.py
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Production source that is offline-by-design.  Tests and the guard script
# itself are excluded (tests legitimately import aiohttp/websockets to exercise
# the local server; the guard only governs shipped code).
OFFLINE_DIRS = [
    "lucy_edge",
    "lucy_core",
    "training",
    "scripts",
]
OFFLINE_FILES = [
    "bridge.py",  # root-level production module
]

# Module roots that mean "can dial out" or pull in an external / cloud model.
# Checked at the dotted-root level.  Local-only plumbing is deliberately absent.
NETWORK_MODULES = {
    "urllib",          # urllib.request/urlopen can fetch; only .parse is benign
    "socket",          # raw sockets can reach anywhere
    "requests",
    "httpx",
    "http",            # http.client
    "telnetlib",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "grpc",
    "paramiko",
    "pycurl",
    # External / cloud model libraries -- never allowed in this ecosystem.
    "openai",
    "anthropic",
    "huggingface",
    "huggingface_hub",
    "transformers",
    "torch",
    "langchain",
    "sklearn",
    "boto3",
    "boto",
    "botocore",
}

# Marker that an external / agent-generated snippet was pasted in, or that an
# export-to-outside path exists.  Such code is forbidden in the ecosystem.
QUARANTINE_MARKERS = (
    "# EXTERNAL_SRC",
    "# @external",
    "EXTERNAL_AGENT_OUTPUT",
    "ONNX",
    "huggingface",
    "upload_model",
    "export_model",
)

# URLs must stay inside the device.  Anything else is an egress attempt.
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_URL_RE = re.compile(r"(?:https?|wss?)://([^\s\"'<>]+)", re.IGNORECASE)


def _imported_modules(tree: ast.Module) -> set[str]:
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


def _is_network(modname: str) -> bool:
    if modname.split(".")[0] not in NETWORK_MODULES:
        return False
    # urllib.parse / urllib.error are parse-only and never dial out.
    if modname in ("urllib.parse", "urllib.error"):
        return False
    return True


def _check_urls(text: str) -> list[str]:
    violations: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0)
        # Skip template / format-string URLs (e.g. "http://{host}:{port}",
        # "http://%s:%s") -- those are bind templates validated at runtime by the
        # phone policy + audit, not hardcoded egress.  We only catch literal,
        # fully-qualified external URLs.
        if "{" in url or "%" in url or "<" in url:
            continue
        authority = m.group(1).split("/", 1)[0]
        host = authority.split("@", 1)[-1].split(":", 1)[0].lower()
        if host not in LOOPBACK_HOSTS:
            violations.append(f"non-loopback URL: {url!r}")
    return violations


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for marker in QUARANTINE_MARKERS:
        if marker in text:
            violations.append(f"quarantine marker {marker!r} present")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        violations.append(f"syntax error: {e}")
        return violations
    for mod in _imported_modules(tree):
        if _is_network(mod):
            violations.append(f"network/cloud import: {mod}")
    violations.extend(_check_urls(text))
    return violations


def _iter_targets() -> list[Path]:
    targets: list[Path] = []
    # Never scan the guard script itself (it defines the quarantine markers).
    guard_self = REPO_ROOT / "scripts" / "ecosystem_guard.py"
    for rel in OFFLINE_DIRS:
        dirp = REPO_ROOT / rel
        if not dirp.exists():
            continue
        for root, _dirs, files in os.walk(dirp):
            for fn in files:
                if fn.endswith(".py"):
                    fp = Path(root) / fn
                    if fp.resolve() == guard_self.resolve():
                        continue
                    targets.append(fp)
    for rel in OFFLINE_FILES:
        fp = REPO_ROOT / rel
        if fp.exists() and fp.is_file():
            targets.append(fp)
    return targets


def main() -> int:
    found: list[tuple[Path, list[str]]] = []
    for fp in _iter_targets():
        v = check_file(fp)
        if v:
            found.append((fp, v))
    if not found:
        scanned = ", ".join(OFFLINE_DIRS + OFFLINE_FILES)
        print(f"[ok] ecosystem_guard: offline paths clean ({scanned})")
        return 0
    print("[FAIL] ecosystem_guard: outbound-capable / non-loopback code in offline-by-design paths:")
    for fp, v in found:
        rel = fp.relative_to(REPO_ROOT)
        for item in v:
            print(f"  {rel}: {item}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
