"""Shared helper: open services around a gateway test."""

from __future__ import annotations

from lucy_edge.services import build_services


async def open_services(config):
    services = build_services(config, fixed_token="test-token")
    await services.open()
    return services
