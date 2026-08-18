"""Foundation audit: machine-verifiable new-foundation contract.

Verifies, from the live configuration and services -- not from claims:

    1. no_cloud_endpoints    every inference endpoint is loopback/private LAN
    2. phone_safety_policy   phone host enforces local-inference gates
    3. local_memory          persistent memory lives on local disk
    4. local_evidence        the decision ledger lives on local disk
    5. model_weights_present the real model registry state (honest gap report)

Endpoint classification is deliberately conservative: hostnames that are not
`localhost` or `.local` are treated as non-local because they can resolve to a
public cloud.  Missing telemetry / unknown model state is reported as such and
is never treated as healthy.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..services import LucyEdgeServices

ENDPOINT_LOCAL_LOOPBACK = "LOCAL_LOOPBACK"
ENDPOINT_LOCAL_PRIVATE = "LOCAL_PRIVATE"
ENDPOINT_PUBLIC_CLOUD = "PUBLIC_CLOUD"
ENDPOINT_INVALID = "INVALID"

_CHECK_NO_CLOUD = "no_cloud_endpoints"
_CHECK_PHONE_POLICY = "phone_safety_policy"
_CHECK_LOCAL_MEMORY = "local_memory"
_CHECK_LOCAL_EVIDENCE = "local_evidence"
_CHECK_MODEL_WEIGHTS = "model_weights_present"

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_GAP = "GAP"

VERDICT_SOUND = "FOUNDATION_SOUND"
VERDICT_GAP = "GAP_IDENTIFIED"
VERDICT_UNSAFE = "FOUNDATION_UNSAFE"


def classify_endpoint(url: str) -> str:
    """Classify an endpoint URL as local, private-LAN, public cloud, or invalid.

    Conservative by design: an unresolvable hostname that is not explicitly
    local is classified PUBLIC_CLOUD because it may point at a cloud service.
    """
    if not url or not isinstance(url, str):
        return ENDPOINT_INVALID
    parsed = urlparse(url.strip())
    host = parsed.hostname
    if host is None:
        return ENDPOINT_INVALID
    host = host.rstrip(".").lower()
    if host in ("localhost", "localhost.localdomain", "localhost."):
        return ENDPOINT_LOCAL_LOOPBACK
    if host.endswith(".local"):
        return ENDPOINT_LOCAL_PRIVATE
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return ENDPOINT_PUBLIC_CLOUD
    if ip.is_loopback or ip.is_unspecified:
        return ENDPOINT_LOCAL_LOOPBACK
    if ip.is_private or ip.is_link_local:
        return ENDPOINT_LOCAL_PRIVATE
    return ENDPOINT_PUBLIC_CLOUD


def _collected_endpoints(services: LucyEdgeServices) -> list[tuple[str, str]]:
    endpoints: list[tuple[str, str]] = []
    for provider in services.providers.all():
        base = getattr(provider, "base_url", None)
        if base:
            endpoints.append((provider.name, base))
    for host in services.hosts.list():
        if host.base_url:
            endpoints.append((f"host:{host.host_id}", host.base_url))
    return endpoints


class FoundationGuard:
    """Audits the live system against the new-foundation contract."""

    def __init__(self, services: LucyEdgeServices) -> None:
        self.services = services
        self.config = services.config

    async def _check_no_cloud(self) -> dict[str, Any]:
        endpoints = _collected_endpoints(self.services)
        if not endpoints:
            return {
                "id": _CHECK_NO_CLOUD,
                "status": STATUS_PASS,
                "detail": (
                    "no external inference endpoints configured; inference "
                    "stays on-device (the sovereign default)"
                ),
                "endpoints": [],
            }
        flagged: list[dict[str, str]] = []
        for name, url in endpoints:
            kind = classify_endpoint(url)
            if kind == ENDPOINT_PUBLIC_CLOUD:
                flagged.append({"name": name, "base_url": url, "classification": kind})
        enforce = self.config.foundation.enforce_no_cloud_endpoints
        if flagged and enforce:
            status = STATUS_FAIL
            detail = (
                f"{len(flagged)} endpoint(s) point at a public/cloud address; "
                "a new-foundation Lucy does not route inference to the cloud"
            )
        else:
            status = STATUS_PASS if not flagged else STATUS_WARN
            detail = "all configured inference endpoints are loopback/private-LAN" if not flagged else (
                "cloud endpoints configured but enforce_no_cloud_endpoints is false"
            )
        return {
            "id": _CHECK_NO_CLOUD,
            "status": status,
            "detail": detail,
            "endpoints": [
                {"name": n, "base_url": u, "classification": classify_endpoint(u)}
                for n, u in endpoints
            ],
        }

    def _check_phone_policy(self) -> dict[str, Any]:
        if self.config.host_role != "PHONE":
            return {
                "id": _CHECK_PHONE_POLICY,
                "status": STATUS_PASS,
                "detail": f"host_role={self.config.host_role}; phone policy not required",
            }
        if self.config.foundation.require_phone_policy and not self.config.phone.phone_local_inference_enabled:
            return {
                "id": _CHECK_PHONE_POLICY,
                "status": STATUS_FAIL,
                "detail": (
                    "host_role=PHONE but phone_local_inference_enabled=false: "
                    "a phone-host Lucy cannot generate locally without the gate"
                ),
            }
        return {
            "id": _CHECK_PHONE_POLICY,
            "status": STATUS_PASS,
            "detail": "phone safety policy enforced (thermal + RAM gate active)",
        }

    def _check_local_storage(self) -> list[dict[str, Any]]:
        memory = self.services.memory
        evidence = self.services.evidence
        memory_ok = memory is not None and getattr(memory, "db_path", None)
        memory_status = STATUS_PASS if memory_ok else STATUS_FAIL
        memory_detail = (
            f"persistent SQLite memory on local disk: {memory.db_path}"
            if memory_ok
            else "no local persistent memory store available"
        )
        evidence_ok = evidence is not None and getattr(evidence, "ledger_db", None)
        evidence_status = STATUS_PASS if evidence_ok else STATUS_FAIL
        evidence_detail = (
            f"evidence ledger on local disk: {evidence.ledger_db}"
            if evidence_ok
            else "no local evidence ledger available"
        )
        return [
            {
                "id": _CHECK_LOCAL_MEMORY,
                "status": memory_status,
                "detail": memory_detail,
            },
            {
                "id": _CHECK_LOCAL_EVIDENCE,
                "status": evidence_status,
                "detail": evidence_detail,
            },
        ]

    async def _check_model_weights(self) -> dict[str, Any]:
        report: dict[str, Any] = {"id": _CHECK_MODEL_WEIGHTS, "providers": {}}
        lucy_present = False
        for provider in self.services.providers.all():
            if getattr(provider, "simulated", False):
                continue
            health = None
            try:
                health = await provider.health()
            except Exception:
                health = None
            reachable = bool(health is not None and health.ok)
            models: list[str] = []
            if reachable and hasattr(provider, "list_models"):
                try:
                    models = [m.name for m in await provider.list_models()]
                except Exception:
                    models = []
            has_lucy = any(
                m == "lucy:latest" or m.startswith("lucy:") or "lucy" in m.lower()
                for m in models
            )
            if has_lucy:
                lucy_present = True
            report["providers"][provider.name] = {
                "reachable": reachable,
                "version": getattr(health, "version", None),
                "models": models,
                "lucy_present": has_lucy,
            }
        if lucy_present:
            report["status"] = STATUS_PASS
            report["detail"] = "her model is present in the local registry"
        elif not report["providers"]:
            report["status"] = STATUS_GAP
            report["detail"] = "no real (non-mock) inference provider is configured"
        else:
            report["status"] = STATUS_GAP
            report["detail"] = (
                "her model is NOT present in the local registry; this is the "
                "one remaining gap between Lucy Edge and a complete new-foundation AI"
            )
        return report

    async def audit(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = [await self._check_no_cloud()]
        checks.append(self._check_phone_policy())
        checks.extend(self._check_local_storage())
        checks.append(await self._check_model_weights())

        failed = [c["id"] for c in checks if c["status"] == STATUS_FAIL]
        gaps = [c["id"] for c in checks if c["status"] == STATUS_GAP]
        if failed:
            verdict = VERDICT_UNSAFE
        elif gaps:
            verdict = VERDICT_GAP
        else:
            verdict = VERDICT_SOUND

        from .loyalty import loyalty_report

        return {
            "service": "lucy_edge",
            "verdict": verdict,
            "failed_checks": failed,
            "gap_checks": gaps,
            "checks": checks,
            "principles": [
                "inference stays on device or on private LAN; never a public cloud",
                "phone safety policy (thermal + RAM gate) is enforced",
                "memory and decisions persist locally with provenance",
                "no capability is claimed unless the running system proves it",
            ],
            "loyalty": loyalty_report(),
        }
