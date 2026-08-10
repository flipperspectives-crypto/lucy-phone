"""Routing policy: the decision system that can ALLOW / DENY / ROUTE / THROTTLE.

Enforces the HARD PHONE POLICY:

    if host_role == PHONE and model_class >= LUCY_7B_CLASS then DENY

plus the global switch ``phone_local_inference_enabled`` (default FALSE).

A routing decision is always machine-readable and is recorded in evidence.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from ..config import LucyEdgeConfig
from ..hardware.governor import GovernorAction, ThermalGovernor
from ..hardware.snapshot import HardwareSnapshot
from .hosts import HostRegistry, HostRole, HostState, HostStatus


class ModelClass(str, Enum):
    UNKNOWN = "UNKNOWN"
    MICRO = "MICRO"  # < 1B
    SMALL = "SMALL"  # 1B - 6.9B
    LUCY_7B_CLASS = "LUCY_7B_CLASS"  # >= 7B


class RoutingDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ROUTE = "ROUTE"
    THROTTLE = "THROTTLE"


class ReasonCode(str, Enum):
    LOCAL_INFERENCE_DISABLED = "LOCAL_INFERENCE_DISABLED"
    PHONE_MODEL_DENIED = "PHONE_MODEL_DENIED"
    UNKNOWN_MODEL = "UNKNOWN_MODEL"
    PROVIDER_OFFLINE = "PROVIDER_OFFLINE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INSUFFICIENT_RAM = "INSUFFICIENT_RAM"
    GPU_TELEMETRY_UNKNOWN = "GPU_TELEMETRY_UNKNOWN"
    THERMAL_LIMIT = "THERMAL_LIMIT"
    THERMAL_UNKNOWN = "THERMAL_UNKNOWN"
    REMOTE_HOST_SELECTED = "REMOTE_HOST_SELECTED"
    NO_REMOTE_HOST = "NO_REMOTE_HOST"
    UNKNOWN_REMOTE_HOST = "UNKNOWN_REMOTE_HOST"
    MOCK_PROVIDER_SELECTED = "MOCK_PROVIDER_SELECTED"
    OK = "OK"
    RATE_LIMITED = "RATE_LIMITED"


# Rough size tags used to classify parameter size.  Explicit and conservative:
# an unparsable name is classified UNKNOWN, never guessed -- unless the operator
# declared a known size for its family (see routing.known_sizes).
_SIZE_RE = re.compile(r"(?i)([0-9]+(?:\.[0-9]+)?)\s*([kmbt])\b")
_ORDER = {"k": 0.000001, "m": 0.001, "b": 1.0, "t": 1000.0}

# Lucy is canonically a 7B-class model in this project.  A bare `lucy` tag with
# no size in the name still means the Lucy 7B build unless the operator declares
# otherwise (e.g. routing.known_sizes: lucy: 1.7 for a phone-quantized build).
_FAMILY_DEFAULT_SIZES_B: dict[str, float] = {"lucy": 7.0}


def _known_family_size(family: str, override: Optional[dict[str, float]]) -> Optional[float]:
    if override and family in override:
        return float(override[family])
    return _FAMILY_DEFAULT_SIZES_B.get(family)


def classify_model(
    name: str,
    known_sizes: Optional[dict[str, float]] = None,
) -> tuple[str, Optional[float], ModelClass]:
    """Return (family, params_in_billions_or_None, model_class).

    Size is read from the name when present; otherwise the operator-declared
    ``known_sizes`` table is consulted; otherwise a known family default (Lucy
    -> 7B) is used; only then is the model UNKNOWN.
    """
    family = name.split(":", 1)[0].strip() if ":" in name else name.strip()
    match = _SIZE_RE.search(name)
    if match:
        value = float(match.group(1))
        unit = _ORDER[match.group(2).lower()]
        params_b = round(value * unit, 2)
    else:
        params_b = _known_family_size(family, known_sizes)
    if params_b is None:
        return (family, None, ModelClass.UNKNOWN)
    if params_b >= 7.0:
        klass = ModelClass.LUCY_7B_CLASS
    elif params_b >= 1.0:
        klass = ModelClass.SMALL
    else:
        klass = ModelClass.MICRO
    return (family, params_b, klass)


_CLASS_RANK = {
    ModelClass.UNKNOWN: 0,
    ModelClass.MICRO: 1,
    ModelClass.SMALL: 2,
    ModelClass.LUCY_7B_CLASS: 3,
}


def class_rank(model_class: ModelClass) -> int:
    return _CLASS_RANK[model_class]


_GIB = 1024 ** 3


def required_ram_bytes(model_class: ModelClass) -> Optional[int]:
    """Conservative free-RAM floor for a model class (phone-local gate)."""
    if model_class == ModelClass.MICRO:
        return int(0.75 * _GIB)
    if model_class == ModelClass.SMALL:
        return 2 * _GIB
    if model_class == ModelClass.LUCY_7B_CLASS:
        return 6 * _GIB
    return None


class RoutingRequest(BaseModel):
    model: str
    provider: str = "mock"
    host_role: HostRole = HostRole.PHONE
    host_id: str = "phone"
    target_host: Optional[str] = None  # explicit requested remote host
    resources: Optional[HardwareSnapshot] = None
    user: Optional[str] = None
    raw: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


class RoutingResult(BaseModel):
    decision: RoutingDecision
    reason_code: ReasonCode
    message: str
    model: str
    model_class: ModelClass
    provider: Optional[str] = None
    target_host: Optional[str] = None
    throttle_seconds: Optional[float] = None
    evidence: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


class RoutingPolicy:
    def __init__(self, config: LucyEdgeConfig) -> None:
        self.config = config
        self.phone_local_inference_enabled = (
            config.phone.phone_local_inference_enabled
        )
        self.known_sizes = dict(config.routing.known_sizes)
        self.governor = ThermalGovernor()
        self.max_local_class = config.phone.local_inference_max_class.upper()
        try:
            self.max_local_class_rank = _CLASS_RANK[ModelClass(self.max_local_class)]
        except ValueError:
            self.max_local_class_rank = _CLASS_RANK[ModelClass.SMALL]
        self.max_ram_utilization = 0.85  # router-level safe ceiling

    def classify(self, name: str) -> tuple[str, Optional[float], ModelClass]:
        return classify_model(name, known_sizes=self.known_sizes)

    def _local_denial(self, request: RoutingRequest, model_class: ModelClass) -> Optional[RoutingResult]:
        """Hard phone policy + global switch, evaluated before anything else."""
        is_local_phone = request.host_role == HostRole.PHONE
        if not is_local_phone:
            return None
        if not self.phone_local_inference_enabled:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.LOCAL_INFERENCE_DISABLED,
                message=(
                    "phone_local_inference_enabled=false: no local model "
                    "generation may occur from Lucy Edge on this phone"
                ),
                model=request.model,
                model_class=model_class,
            )
        if class_rank(model_class) > self.max_local_class_rank:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.PHONE_MODEL_DENIED,
                message=(
                    f"host_role=PHONE and model_class={model_class.value} exceeds "
                    f"phone local_inference_max_class={self.max_local_class}: "
                    "this model class is never run locally on the phone"
                ),
                model=request.model,
                model_class=model_class,
            )
        return None

    def phone_local_gate(self, request: RoutingRequest) -> Optional[RoutingResult]:
        """Thermal + RAM gate for phone-local inference.

        Returns a denial/throttle result when the phone cannot be verified safe
        to run the model locally, or None when local inference may proceed.
        Missing telemetry is never treated as cool (v2.2.9 lesson).
        """
        _, _, model_class = self.classify(request.model)
        snapshot = request.resources
        if snapshot is None or not snapshot.telemetry_available:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.THERMAL_UNKNOWN,
                message=(
                    "no verified thermal/telemetry state on this phone; "
                    "phone-local inference refused"
                ),
                model=request.model,
                model_class=model_class,
                provider=request.provider,
            )
        gov = self.governor.evaluate(snapshot)
        if gov.action == GovernorAction.UNKNOWN:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.THERMAL_UNKNOWN,
                message=f"phone temperature unknown: {gov.reason}",
                model=request.model,
                model_class=model_class,
                provider=request.provider,
            )
        if gov.action == GovernorAction.STOP:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.THERMAL_LIMIT,
                message=gov.reason,
                model=request.model,
                model_class=model_class,
                provider=request.provider,
            )
        if gov.action == GovernorAction.THROTTLE:
            return RoutingResult(
                decision=RoutingDecision.THROTTLE,
                reason_code=ReasonCode.THERMAL_LIMIT,
                message=gov.reason,
                model=request.model,
                model_class=model_class,
                provider=request.provider,
                throttle_seconds=gov.cooldown_seconds,
            )
        required = required_ram_bytes(model_class)
        if (
            required is not None
            and snapshot.ram_available_bytes is not None
            and snapshot.ram_available_bytes < required
        ):
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.INSUFFICIENT_RAM,
                message=(
                    f"phone has {snapshot.ram_available_bytes} bytes available "
                    f"but {required} required for model_class={model_class.value}"
                ),
                model=request.model,
                model_class=model_class,
                provider=request.provider,
            )
        return None

    async def evaluate(self, request: RoutingRequest) -> RoutingResult:
        _, params_b, model_class = self.classify(request.model)

        # Mock provider is the phone-safe fallback for tests only; it performs
        # NO real inference, so the phone-local-inference denial does not apply.
        if request.provider == "mock":
            if not self.config.routing.allow_mock_generation:
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.MOCK_PROVIDER_SELECTED,
                    message="mock generation disabled by routing.allow_mock_generation",
                    model=request.model,
                    model_class=model_class,
                    provider="mock",
                )
            return RoutingResult(
                decision=RoutingDecision.ALLOW,
                reason_code=ReasonCode.MOCK_PROVIDER_SELECTED,
                message="mock provider selected (synthetic; no real inference)",
                model=request.model,
                model_class=model_class,
                provider="mock",
            )

        local_denial = self._local_denial(request, model_class)
        if local_denial is not None:
            return local_denial

        # Unknown model class -> deny by default (never guess).
        if model_class == ModelClass.UNKNOWN:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.UNKNOWN_MODEL,
                message=f"unable to classify model '{request.model}'; refusing to guess",
                model=request.model,
                model_class=model_class,
            )

        return RoutingResult(
            decision=RoutingDecision.ALLOW,
            reason_code=ReasonCode.OK,
            message="policy allows this request",
            model=request.model,
            model_class=model_class,
            provider=request.provider,
        )


def resolve_host_role(raw: str) -> HostRole:
    for role in HostRole:
        if role.value == raw.upper():
            return role
    return HostRole.UNKNOWN


def build_remote_only_result(
    request: RoutingRequest, reason: ReasonCode, message: str, host: Optional[HostState]
) -> RoutingResult:
    _, _, model_class = classify_model(request.model)
    return RoutingResult(
        decision=RoutingDecision.DENY,
        reason_code=reason,
        message=message,
        model=request.model,
        model_class=model_class,
        provider=request.provider,
        target_host=host.host_id if host else request.target_host,
    )
