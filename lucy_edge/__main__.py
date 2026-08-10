"""NEXUS LUCY EDGE CLI.

    python -m lucy_edge introspect [--config lucy_edge.yaml] [--base-dir .]
    python -m lucy_edge serve [--config lucy_edge.yaml] [--base-dir .]
                              [--host HOST] [--port PORT]

Phone-safe: uses the mock provider and performs no model inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from aiohttp import web

from .config import load_config
from .gateway.server import create_app
from .services import build_services


def _discover_config(base_dir: str) -> str | None:
    for name in ("lucy_edge.yaml", "lucy_edge.yml"):
        candidate = Path(base_dir) / name
        if candidate.exists():
            return str(candidate)
    return None


async def _introspect(args: argparse.Namespace) -> int:
    config = load_config(args.config or _discover_config(args.base_dir), base_dir=args.base_dir)
    services = build_services(config)
    await services.open()
    try:
        report = await services.introspection.report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        await services.close()


async def _foundation(args: argparse.Namespace) -> int:
    config = load_config(args.config or _discover_config(args.base_dir), base_dir=args.base_dir)
    services = build_services(config)
    await services.open()
    try:
        audit = await services.foundation.audit()
        if getattr(args, "audit", False):
            print(json.dumps(audit, indent=2, sort_keys=True))
        else:
            print(
                f"verdict: {audit['verdict']}"
                f"  failed={audit['failed_checks'] or 'none'}"
                f"  gaps={audit['gap_checks'] or 'none'}"
            )
            for check in audit["checks"]:
                print(f"  [{check['status']:<4}] {check['id']}: {check['detail']}")
        return 0
    finally:
        await services.close()


async def _serve(args: argparse.Namespace) -> int:
    config = load_config(args.config or _discover_config(args.base_dir), base_dir=args.base_dir)
    host = args.host or config.gateway.host
    port = args.port or config.gateway.port
    services = build_services(config)
    await services.open()
    app = create_app(services)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(
        f"lucy_edge gateway listening on http://{host}:{port} "
        f"(auth_enabled={config.gateway.auth_enabled}, token source: {services.auth.source})"
    )
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
        await services.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lucy_edge", description="NEXUS LUCY EDGE")
    parser.add_argument("--config", default=None, help="path to a lucy_edge.yaml config")
    parser.add_argument("--base-dir", default=".", help="project base directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("introspect", help="print the runtime introspection report")
    foundation = sub.add_parser(
        "foundation",
        help="audit the new-foundation contract (no cloud, local memory/evidence, model gap)",
    )
    foundation.add_argument("--audit", action="store_true", help="print the full audit JSON")
    serve = sub.add_parser("serve", help="run the control-plane gateway HTTP server")
    serve.add_argument("--host", default=None, help="bind address (default: config gateway.host)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default: config gateway.port)")
    args = parser.parse_args(argv)

    if args.command == "introspect":
        return asyncio.run(_introspect(args))
    if args.command == "foundation":
        return asyncio.run(_foundation(args))
    if args.command == "serve":
        return asyncio.run(_serve(args))
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
