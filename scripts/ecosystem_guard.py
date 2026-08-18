#!/usr/bin/env python3
"""Ecosystem guard: enforce the strictly-offline rule on offline-by-design paths.

Scans a fixed set of in-repo paths that MUST NEVER make outbound network calls.
Fails (exit 1) if any Python file there imports a network module or carries a
quarantine marker comment. This is the technical backstop for Lucy's
fail-closed / no-public-cloud directive.

Usage:  python3 scripts/ecosystem_guard.py
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths that are offline-by-design. Missing paths are skipped.
OFFLINE_PATHS = [
    "training",
    "lucy_core",
    "lucy_edge/introspection",
]

# Module roots that mean "can dial out". Checked at the dotted-root level.
NETWORK_MODULES = {
    "urllib",
    "socket",
    "requests",
    "httpx",
    "http",
    "telnetlib",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "aiohttp",
    "websocket",
    "websockets",
    "grpc",
    "paramiko",
    "pycurl",
}

# Marker that an external / agent-generated snippet was pasted in. Such code is
# forbidden in the ecosystem; the guard catches it if a marker is left behind.
QUARANTINE_MARKERS = ("# EXTERNAL_SRC", "# @external", "EXTERNAL_AGENT_OUTPUT")


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
    return modname.split(".")[0] in NETWORK_MODULES


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
            violations.append(f"network import: {mod}")
    return violations


def main() -> int:
    found: list[tuple[Path, list[str]]] = []
    for rel in OFFLINE_PATHS:
        dirp = REPO_ROOT / rel
        if not dirp.exists():
            continue
        for root, _dirs, files in os.walk(dirp):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                fp = os.path.join(root, fn)
                v = check_file(Path(fp))
                if v:
                    found.append((Path(fp), v))
    if not found:
        print(f"[ok] ecosystem_guard: offline paths clean ({', '.join(OFFLINE_PATHS)})")
        return 0
    print("[FAIL] ecosystem_guard: outbound-capable code in offline-by-design paths:")
    for fp, v in found:
        rel = fp.relative_to(REPO_ROOT)
        for item in v:
            print(f"  {rel}: {item}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
