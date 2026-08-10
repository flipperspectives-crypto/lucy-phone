"""Thermal governor.

Encodes the lesson from the v2.2.9 thermal incident: thermal state is checked
BEFORE retry, a critical thermal state never initiates inference, and no
mid-inference cancellation is ever claimed as proven.  Pure decision logic;
never starts or stops model processes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .snapshot import HardwareSnapshot


class GovernorAction(str, Enum):
    NORMAL = "NORMAL"
    THROTTLE = "THROTTLE"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"


class ThermalLimits(BaseModel):
    warn_c: float = 65.0
    critical_c: float = 75.0
    hard_stop_c: float = 85.0


class GovernanceResult(BaseModel):
    action: GovernorAction
    reason: Optional[str] = None
    observed_temperature_c: Optional[float] = None
    cooldown_seconds: Optional[float] = None
    abort_seen: bool = False
    cancellation_proven: bool = False
    details: list[str] = Field(default_factory=list)


class ThermalGovernor:
    def __init__(self, limits: Optional[ThermalLimits] = None) -> None:
        self.limits = limits or ThermalLimits()

    def evaluate(self, snapshot: HardwareSnapshot) -> GovernanceResult:
        """Evaluate a snapshot against thermal limits.

        Returns a decision only.  Missing temperature is UNKNOWN, not 0, and
        must not be treated as cool.
        """
        temp = snapshot.cpu_temperature_c
        source = snapshot.thermal_source
        details: list[str] = []
        if temp is None:
            return GovernanceResult(
                action=GovernorAction.UNKNOWN,
                reason="no CPU temperature telemetry available",
                observed_temperature_c=None,
                abort_seen=False,
                cancellation_proven=False,
                details=["temperature unknown: no claim of safe thermal state"],
            )

        details.append(f"source={source or 'unknown'}")
        if temp >= self.limits.hard_stop_c:
            return GovernanceResult(
                action=GovernorAction.STOP,
                reason=f"temperature {temp}C >= hard_stop {self.limits.hard_stop_c}C",
                observed_temperature_c=temp,
                cooldown_seconds=30.0,
                abort_seen=True,
                cancellation_proven=False,
                details=details,
            )
        if temp >= self.limits.critical_c:
            return GovernanceResult(
                action=GovernorAction.STOP,
                reason=f"temperature {temp}C >= critical {self.limits.critical_c}C",
                observed_temperature_c=temp,
                cooldown_seconds=15.0,
                abort_seen=True,
                cancellation_proven=False,
                details=details,
            )
        if temp >= self.limits.warn_c:
            return GovernanceResult(
                action=GovernorAction.THROTTLE,
                reason=f"temperature {temp}C >= warn {self.limits.warn_c}C",
                observed_temperature_c=temp,
                cooldown_seconds=5.0,
                abort_seen=False,
                cancellation_proven=False,
                details=details,
            )
        return GovernanceResult(
            action=GovernorAction.NORMAL,
            reason=f"temperature {temp}C below warn {self.limits.warn_c}C",
            observed_temperature_c=temp,
            abort_seen=False,
            cancellation_proven=False,
            details=details,
        )

    def before_retry(self, snapshot: HardwareSnapshot) -> GovernanceResult:
        """Thermal re-check before any retry (per evolution retry rule)."""
        result = self.evaluate(snapshot)
        if result.action in (GovernorAction.STOP, GovernorAction.THROTTLE):
            result.details.append(
                "retry blocked: thermal state not safe (server-side mid-inference "
                "cancellation is NOT proven)"
            )
        return result
