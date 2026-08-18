"""Builtin tool implementations (system / model / memory / evidence / files / git)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Optional

from ...evidence.schema import EvidenceRecord, EvidenceType
from ...providers.base import Capability
from ..context import ToolContext
from ..registry import ToolRegistry, ToolSpec

_MAX_READ_BYTES = 64 * 1024
_MAX_WRITE_BYTES = 256 * 1024
_GIT_TIMEOUT = 10.0


async def _system_health(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    memory_count = (
        await context.memory_store.count() if context.memory_store is not None else None
    )
    evidence_count = (
        await context.evidence.count() if context.evidence is not None else None
    )
    hosts = context.hosts.list() if context.hosts is not None else []
    provider_summary = (
        context.providers.summary() if context.providers is not None else {}
    )
    return {
        "service": "lucy_edge",
        "host_role": context.config.host_role if context.config else None,
        "host_id": context.config.host_id if context.config else None,
        "memory_records": memory_count,
        "evidence_records": evidence_count,
        "remote_hosts": [],
        "providers": provider_summary,
        "phone_local_inference_enabled": (
            context.config.phone.phone_local_inference_enabled if context.config else None
        ),
    }


async def _system_capabilities(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    if context.introspection is None:
        return {"error": "introspection service unavailable"}
    return await context.introspection.capabilities_report()


async def _model_list(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    out: dict[str, Any] = {}
    if context.providers is None:
        return {"error": "provider registry unavailable"}
    for name in context.providers.names():
        provider = context.providers.get(name)
        try:
            if provider.supports(Capability.LIST_MODELS):
                models = await provider.list_models()
                out[name] = [m.model_dump() for m in models]
            else:
                out[name] = "unavailable"
        except Exception as exc:
            out[name] = f"error: {type(exc).__name__}"
    return out


async def _model_health(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    provider_name = kw.get("provider", "mock")
    if context.providers is None:
        return {"error": "provider registry unavailable"}
    provider = context.providers.get(provider_name)
    if provider is None:
        return {"provider": provider_name, "ok": False, "error": "unknown provider"}
    health = await provider.health()
    return health.model_dump()


async def _model_route(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    model = kw.get("model")
    if not model:
        return {"error": "model is required"}
    provider = kw.get("provider", "mock")
    target_host = kw.get("target_host")
    if context.router is None:
        return {"error": "router unavailable"}
    from ...routing.policy import RoutingRequest
    from ...routing.hosts import HostRole

    request = RoutingRequest(
        model=model,
        provider=provider,
        host_role=HostRole(context.config.host_role) if context.config else HostRole.UNKNOWN,
        host_id=context.config.host_id if context.config else "unknown",
        target_host=target_host,
    )
    result = await context.router.route(request)
    if context.evidence is not None:
        record = EvidenceRecord(
            record_type=EvidenceType.ROUTING_DECISION,
            goal=f"routing decision for {model}",
            model=result.model,
            provider=result.provider,
            host_role=request.host_role.value,
            routing_decision=result.decision.value,
            routing_reason_code=result.reason_code.value,
            host=request.host_id,
            completion_reason=result.message,
        )
        await context.evidence.append(record)
        result.evidence["run_id"] = record.run_id
    return {
        "decision": result.decision.value,
        "reason_code": result.reason_code.value,
        "message": result.message,
        "model_class": result.model_class.value,
        "provider": result.provider,
        "target_host": result.target_host,
        "evidence_run_id": result.evidence.get("run_id"),
    }


async def _memory_search(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    query = kw.get("query", "")
    limit = int(kw.get("limit", 8))
    if context.retrieval is None:
        return {"error": "memory unavailable"}
    results = await context.retrieval.retrieve(query, limit=limit)
    return {
        "count": len(results),
        "results": [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "memory_type": r.memory_type.value,
                "provenance": r.provenance.value,
                "status": r.status.value,
                "confidence": r.confidence,
                "score": score,
            }
            for r, score in results
        ],
    }


async def _memory_write_proposal(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    content = kw.get("content")
    if not content:
        return {"error": "content is required"}
    if context.admission is None:
        return {"error": "memory admission unavailable"}
    record = await context.admission.suggest_from_model(
        content=content,
        source=kw.get("source", "model_output"),
        project=kw.get("project"),
        metadata=kw.get("metadata"),
    )
    return {
        "memory_id": record.memory_id,
        "status": record.status.value,
        "provenance": record.provenance.value,
        "sha256": record.sha256,
        "note": "proposed memory is not durable truth until reviewed",
    }


async def _memory_inspect(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    memory_id = kw.get("memory_id")
    if memory_id:
        if context.memory_store is None:
            return {"error": "memory unavailable"}
        record = await context.memory_store.get(memory_id)
        if record is None:
            return {"error": "no such memory"}
        return record.model_dump()
    limit = int(kw.get("limit", 10))
    if context.memory_store is None:
        return {"error": "memory unavailable"}
    records = await context.memory_store.list(limit=limit)
    return {"count": len(records), "records": [r.model_dump() for r in records]}


async def _evidence_query(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    if context.evidence is None:
        return {"error": "evidence ledger unavailable"}
    limit = int(kw.get("limit", 20))
    record_type = kw.get("record_type")
    from ...evidence.schema import EvidenceType

    records = await context.evidence.query(
        record_type=EvidenceType(record_type) if record_type else None,
        limit=limit,
    )
    return {"count": len(records), "records": records}


async def _files_read_scoped(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    path = kw.get("path")
    if not path:
        return {"error": "path is required"}
    target = Path(path)
    if not target.is_file():
        return {"error": "not a file"}
    data = target.read_bytes()[: _MAX_READ_BYTES]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "binary file; refusing to return raw bytes"}
    return {"path": str(target), "size_bytes": len(data), "content": text}


async def _files_write_scoped(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    path = kw.get("path")
    content = kw.get("content", "")
    if not path:
        return {"error": "path is required"}
    data = content.encode("utf-8")
    if len(data) > _MAX_WRITE_BYTES:
        return {"error": f"content exceeds {_MAX_WRITE_BYTES} bytes"}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": str(target), "size_bytes": len(data), "ok": True}


async def _files_delete_scoped(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    path = kw.get("path")
    if not path:
        return {"error": "path is required"}
    target = Path(path)
    if not target.exists():
        return {"error": "path does not exist"}
    target.unlink()
    return {"path": str(target), "deleted": True}


async def _git_status(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    return await _run_git(["status", "--short", "--branch"], context.workspace)


async def _git_diff(**kw: Any) -> dict[str, Any]:
    context: ToolContext = kw["context"]
    return await _run_git(["diff", "--stat"], context.workspace)


async def _shell_exec_denied(**kw: Any) -> dict[str, Any]:
    # Unreachable: permission policy denies shell.* before execution.
    return {"error": "arbitrary shell is denied"}


async def _run_git(argv: list[str], cwd: str) -> dict[str, Any]:
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "git",
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=_GIT_TIMEOUT,
        )
        stdout, stderr = await proc.communicate()
        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", "replace")[:8000],
            "stderr": stderr.decode("utf-8", "replace")[:2000],
        }
    except asyncio.TimeoutError:
        return {"error": f"git timed out after {_GIT_TIMEOUT}s"}
    except FileNotFoundError:
        return {"error": "git executable not found"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def register_builtin_tools(registry: ToolRegistry, context: ToolContext) -> None:
    tools: list[ToolSpec] = [
        ToolSpec("system.health", "runtime health summary", _system_health, "read"),
        ToolSpec(
            "system.capabilities",
            "verified runtime capability report",
            _system_capabilities,
            "read",
        ),
        ToolSpec("model.list", "list models per provider", _model_list, "read"),
        ToolSpec("model.health", "provider health", _model_health, "read"),
        ToolSpec("model.route", "routing decision for a model", _model_route, "read"),
        ToolSpec("memory.search", "retrieve relevant memory", _memory_search, "read"),
        ToolSpec(
            "memory.write_proposal",
            "propose a memory (non-durable until reviewed)",
            _memory_write_proposal,
            "read",
        ),
        ToolSpec("memory.inspect", "inspect memory record(s)", _memory_inspect, "read"),
        ToolSpec("evidence.query", "query the evidence ledger", _evidence_query, "read"),
        ToolSpec("files.read_scoped", "read a file inside approved roots", _files_read_scoped, "read"),
        ToolSpec(
            "files.write_scoped", "write a file inside approved roots", _files_write_scoped, "write"
        ),
        ToolSpec(
            "files.delete_scoped", "delete a file inside approved roots", _files_delete_scoped, "delete"
        ),
        ToolSpec("git.status", "non-destructive git status", _git_status, "git_read"),
        ToolSpec("git.diff", "non-destructive git diff", _git_diff, "git_read"),
        ToolSpec(
            "shell.exec",
            "arbitrary shell execution (always denied)",
            _shell_exec_denied,
            "shell",
        ),
    ]
    for spec in tools:
        registry.register(spec)
