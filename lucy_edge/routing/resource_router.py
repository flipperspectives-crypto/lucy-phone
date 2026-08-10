"""Hardware-aware host selection.

Pure decision logic: given candidate host snapshots and host roles, choose a
preferred host.  Never fabricates telemetry; missing values are None/UNKNOWN
and are reported as such.  This module is read-only and never initiates
inference.
"""

from __future__ import annotations

from typing import Optional

from ..hardware.snapshot import HardwareSnapshot
from .hosts import HostRole, HostState
from .policy import ReasonCode, RoutingDecision, RoutingResult


class HostCandidate:
    def __init__(self, host: HostState, snapshot: Optional[HardwareSnapshot]) -> None:
        self.host = host
        self.snapshot = snapshot

    @property
    def free_ram_bytes(self) -> Optional[int]:
        if self.snapshot is not None:
            return self.snapshot.ram_available_bytes
        return self.host.ram_free_bytes

    @property
    def has_gpu(self) -> bool:
        if self.snapshot is not None and self.snapshot.gpu_name:
            return True
        return bool(self.host.gpu_name)


def select_host(candidates: list[HostCandidate]) -> Optional[HostCandidate]:
    """Pick the best host from candidates.

    Ranking (heuristic, explicit):
      1. usable hosts only
      2. prefer a host with a GPU
      3. prefer a host with more free RAM
      4. deterministic tie-break by host_id
    """
    usable = [c for c in candidates if c.host.is_usable]
    if not usable:
        return None
    scored: list[tuple[float, str, HostCandidate]] = []
    for cand in usable:
        gpu_score = 1.0 if cand.has_gpu else 0.0
        ram = cand.free_ram_bytes
        ram_score = 0.0 if ram is None else min(ram / (16 * 1024**3), 1.0)
        score = gpu_score * 2.0 + ram_score
        scored.append((score, cand.host.host_id, cand))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def resource_gate(
    host: HostState,
    snapshot: Optional[HardwareSnapshot],
    required_ram_bytes: Optional[int] = None,
) -> RoutingResult:
    """Gate a host by available resources (RAM / GPU telemetry knowledge)."""
    if snapshot is not None and snapshot.telemetry_available:
        available = snapshot.ram_available_bytes
        if available is not None and required_ram_bytes is not None:
            if available < required_ram_bytes:
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.INSUFFICIENT_RAM,
                    message=(
                        f"host '{host.host_id}' has {available} bytes available "
                        f"but {required_ram_bytes} required"
                    ),
                    model="",
                    model_class="UNKNOWN",
                    provider=host.provider,
                    target_host=host.host_id,
                )
        if snapshot.gpu_name is None:
            return RoutingResult(
                decision=RoutingDecision.THROTTLE,
                reason_code=ReasonCode.GPU_TELEMETRY_UNKNOWN,
                message=(
                    f"host '{host.host_id}' reports telemetry but no GPU identity; "
                    "routing is throttled until GPU telemetry is known"
                ),
                model="",
                model_class="UNKNOWN",
                provider=host.provider,
                target_host=host.host_id,
                throttle_seconds=5.0,
            )
        return RoutingResult(
            decision=RoutingDecision.ALLOW,
            reason_code=ReasonCode.OK,
            message=f"host '{host.host_id}' passes the resource gate",
            model="",
            model_class="UNKNOWN",
            provider=host.provider,
            target_host=host.host_id,
        )
    return RoutingResult(
        decision=RoutingDecision.THROTTLE,
        reason_code=ReasonCode.GPU_TELEMETRY_UNKNOWN,
        message=(
            f"host '{host.host_id}' has no telemetry available; cannot verify "
            "resources. Throttling until telemetry exists."
        ),
        model="",
        model_class="UNKNOWN",
        provider=host.provider,
        target_host=host.host_id,
        throttle_seconds=5.0,
    )
