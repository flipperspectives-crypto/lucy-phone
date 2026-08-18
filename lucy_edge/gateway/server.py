"""NEXUS LUCY EDGE gateway HTTP server (aiohttp).

Control plane only.  No model weights, no Ollama startup, no inference
initiation beyond the routing gated provider call (mock in phone phase).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from aiohttp import web

from ..agent.runtime import AgentRuntime
from ..api.schemas import ChatRequest
from ..evidence.schema import EvidenceType
from ..providers.base import Capability
from ..routing.hosts import HostState
from ..routing.policy import RoutingDecision, RoutingRequest
from ..version import __version__
from ..services import LucyEdgeServices

SERVICES_KEY = web.AppKey("lucy_edge_services", LucyEdgeServices)
TASKS_KEY: web.AppKey = web.AppKey(
    "lucy_edge_tasks", dict[str, dict[str, Any]]
)


def create_app(services: LucyEdgeServices) -> web.Application:
    app = web.Application()
    app[SERVICES_KEY] = services
    app[TASKS_KEY] = {}  # run_id -> {"runtime": AgentRuntime, "task": asyncio.Task, "result": None}
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/", _handle_index)
    app.router.add_get("/v1/auth/status", _handle_auth_status)
    app.router.add_post("/v1/chat", _handle_chat)
    app.router.add_post("/v1/chat/stream", _handle_chat_stream)
    app.router.add_post("/v1/agent/tasks", _handle_agent_submit)
    app.router.add_get("/v1/agent/tasks/{run_id}", _handle_agent_status)
    app.router.add_get("/v1/evidence", _handle_evidence)
    app.router.add_get("/v1/lucy/introspect", _handle_introspect)
    app.router.add_get("/v1/foundation", _handle_foundation)
    app.router.add_get("/v1/grounding", _handle_grounding)
    app.router.add_get("/v1/hardware/snapshot", _handle_hardware)
    return app


def _services(request: web.Request) -> LucyEdgeServices:
    return request.app[SERVICES_KEY]


async def _guard(request: web.Request) -> Optional[web.Response]:
    services = _services(request)
    if not services.auth.check(request.headers.get("Authorization")):
        return web.json_response(
            {"error": "unauthorized"}, status=401, headers={"WWW-Authenticate": "Bearer"}
        )
    decision = services.rate_limiter.allow(request.remote or "unknown")
    if not decision.allowed:
        return web.json_response(
            {"error": "rate limit exceeded", "retry_after": decision.retry_after},
            status=429,
            headers={"Retry-After": str(int(decision.retry_after or 0))},
        )
    return None


async def _handle_health(request: web.Request) -> web.StreamResponse:
    services = _services(request)
    memory_count = await services.memory.count() if services.memory else None
    evidence_count = await services.evidence.count() if services.evidence else None
    return web.json_response(
        {
            "status": "ok",
            "service": "lucy_edge",
            "version": __version__,
            "host_role": services.config.host_role,
            "host_id": services.config.host_id,
            "workspace_exists": True,
            "memory_records": memory_count,
            "evidence_records": evidence_count,
        }
    )


async def _handle_index(request: web.Request) -> web.StreamResponse:
    services = _services(request)
    html = (
        _INDEX_PAGE
        .replace("__VERSION__", __version__)
        .replace("__HOST_ROLE__", services.config.host_role)
        .replace("__HOST_ID__", services.config.host_id)
    )
    return web.Response(text=html, content_type="text/html")


_INDEX_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEXUS LUCY EDGE</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 720px;
         padding: 1.5rem; background: #101418; color: #e6edf3; }
  h1 { margin-bottom: .25rem; }
  .meta { color: #8b949e; font-size: .9rem; }
  input, select, textarea, button { font: inherit; padding: .5rem; margin: .25rem 0;
         border-radius: 8px; border: 1px solid #30363d; background: #161b22;
         color: #e6edf3; width: 100%; box-sizing: border-box; }
  button { background: #1f6feb; border-color: #1f6feb; cursor: pointer; font-weight: 600; }
  button:hover { background: #388bfd; }
  #health, #out { background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
         padding: .75rem; margin-top: 1rem; white-space: pre-wrap; word-break: break-word; }
  .row { display: flex; gap: .5rem; }
  .row > * { flex: 1; }
</style>
</head>
<body>
<h1>NEXUS LUCY EDGE</h1>
<div class="meta">version __VERSION__ &middot; host_role __HOST_ROLE__ &middot; host_id __HOST_ID__ &middot; NOT PRODUCTION-READY</div>

<label>Bearer token <input id="tok" type="password" placeholder="data/operator.token value" autocomplete="off"></label>
<div id="health">loading health&hellip;</div>
<div id="foundation">loading foundation audit&hellip;</div>

<h3>Chat</h3>
<div class="row">
  <select id="provider"><option selected>local_lucy</option><option>mock</option></select>
  <input id="model" value="lucy:latest" placeholder="model">
</div>
<textarea id="msg" rows="2" placeholder="message"></textarea>
<button onclick="chat()">Send</button>

<h3>Grounding &mdash; what she knows locally</h3>
<button onclick="ground()">Ground current message</button>

<h3>Introspect</h3>
<button onclick="introspect()">Show capability report</button>
<button onclick="foundation()">Show full foundation audit</button>

<div id="out">response will appear here</div>

<script>
const tok = () => { const t = document.getElementById('tok').value; localStorage.setItem('tok', t); return t; };
document.getElementById('tok').value = localStorage.getItem('tok') || '';
async function call(path, body) {
  const r = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + tok() },
    body: body ? JSON.stringify(body) : undefined,
  });
  return JSON.stringify(await r.json(), null, 2);
}
async function chat() {
  document.getElementById('out').textContent = 'waiting...';
  try {
    const body = {
      model: document.getElementById('model').value,
      provider: document.getElementById('provider').value,
      messages: [{ role: 'user', content: document.getElementById('msg').value }],
    };
    document.getElementById('out').textContent = await call('/v1/chat', body);
  } catch (e) { document.getElementById('out').textContent = String(e); }
}
async function introspect() {
  document.getElementById('out').textContent = 'waiting...';
  try { document.getElementById('out').textContent = await call('/v1/lucy/introspect'); }
  catch (e) { document.getElementById('out').textContent = String(e); }
}
async function foundation() {
  document.getElementById('out').textContent = 'auditing...';
  try { document.getElementById('out').textContent = await call('/v1/foundation'); }
  catch (e) { document.getElementById('out').textContent = String(e); }
}
async function ground() {
  const q = document.getElementById('msg').value.trim();
  if (!q) { document.getElementById('out').textContent = 'type a question first'; return; }
  document.getElementById('out').textContent = 'grounding from her local records...';
  try { document.getElementById('out').textContent = await call('/v1/grounding?q=' + encodeURIComponent(q)); }
  catch (e) { document.getElementById('out').textContent = String(e); }
}
fetch('/health').then(r => r.json()).then(h => {
  document.getElementById('health').textContent =
    'status: ' + h.status + ' | role: ' + h.host_role + ' | memory: ' + h.memory_records +
    ' | evidence: ' + h.evidence_records;
}).catch(() => { document.getElementById('health').textContent = 'health unreachable'; });
fetch('/v1/foundation', { headers: { 'Authorization': 'Bearer ' + tok() } }).then(r => r.json()).then(a => {
  document.getElementById('foundation').textContent =
    'verdict: ' + a.verdict + ' | checks: ' + (a.checks || []).map(c => c.status).join('/');
}).catch(() => { document.getElementById('foundation').textContent = 'foundation audit unavailable (token needed)'; });
</script>
</body>
</html>
"""


async def _handle_auth_status(request: web.Request) -> web.StreamResponse:
    services = _services(request)
    return web.json_response(services.auth.describe())


async def _handle_chat(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    try:
        raw = await request.json()
        chat = ChatRequest.model_validate(raw)
    except Exception as exc:
        return web.json_response({"error": f"invalid request: {exc}"}, status=422)

    routing_request = RoutingRequest(
        model=chat.model,
        provider=chat.provider,
        host_role=services.config.host_role,
        host_id=services.config.host_id,
        target_host=chat.target_host,
        resources=services.telemetry.snapshot(),
    )
    result = await services.router.route(routing_request)
    evidence_run_id = await services.record_routing(routing_request, result)

    if result.decision == RoutingDecision.DENY:
        return web.json_response(
            {
                "ok": False,
                "routing": result.model_dump(),
                "evidence_run_id": evidence_run_id,
                "error": result.message,
            }
        )

    if result.decision == RoutingDecision.ROUTE:
        return web.json_response(
            {
                "ok": False,
                "routing": result.model_dump(),
                "evidence_run_id": evidence_run_id,
                "error": (
                    f"remote inference routed to '{result.target_host}' but remote "
                    "chat is not connected in the phone phase"
                ),
            }
        )

    provider = services.providers.get(result.provider or chat.provider)
    if provider is None:
        return web.json_response(
            {"ok": False, "error": f"provider '{result.provider}' not available"}
        )
    try:
        response = await provider.chat(chat.messages, model=chat.model, num_predict=chat.max_tokens)
    except Exception as exc:
        return web.json_response(
            {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        )
    return web.json_response(
        {
            "ok": True,
            "routing": result.model_dump(),
            "evidence_run_id": evidence_run_id,
            "provider": response.provider,
            "model": response.model,
            "message": response.message,
            "simulated": response.simulated,
        }
    )


async def _handle_chat_stream(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    try:
        raw = await request.json()
        chat = ChatRequest.model_validate(raw)
    except Exception as exc:
        return web.json_response({"error": f"invalid request: {exc}"}, status=422)

    routing_request = RoutingRequest(
        model=chat.model,
        provider=chat.provider,
        host_role=services.config.host_role,
        host_id=services.config.host_id,
        target_host=chat.target_host,
        resources=services.telemetry.snapshot(),
    )
    result = await services.router.route(routing_request)

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)

    def _event(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    if result.decision != RoutingDecision.ALLOW:
        await resp.write(_event({"type": "error", "message": result.message, "routing": result.model_dump()}))
        await resp.write_eof()
        return resp

    provider = services.providers.get(result.provider or chat.provider)
    if provider is None or not provider.supports(Capability.STREAM_CHAT):
        await resp.write(_event({"type": "error", "message": "streaming unavailable for this provider"}))
        await resp.write_eof()
        return resp
    try:
        async for chunk in provider.stream_chat(chat.messages, model=chat.model):
            await resp.write(_event({"type": "chunk", "text": chunk.text, "done": chunk.done}))
            if chunk.done:
                break
    except Exception as exc:
        await resp.write(_event({"type": "error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"}))
    await resp.write_eof()
    return resp


async def _handle_agent_submit(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    try:
        raw = await request.json()
        goal = str(raw.get("goal", ""))
        if not goal:
            return web.json_response({"error": "goal is required"}, status=422)
        limits = raw.get("limits")
    except Exception as exc:
        return web.json_response({"error": f"invalid request: {exc}"}, status=422)

    runtime = services.new_agent_run(goal, limits=limits)
    task = asyncio.get_running_loop().create_task(_run_agent(runtime, request.app))
    request.app[TASKS_KEY][runtime.run_id] = {
        "runtime": runtime,
        "task": task,
        "result": None,
    }
    return web.json_response(
        {
            "run_id": runtime.run_id,
            "goal": goal,
            "state": runtime.state.value,
            "status_url": f"/v1/agent/tasks/{runtime.run_id}",
        },
        status=202,
    )


async def _run_agent(runtime: AgentRuntime, app: web.Application) -> None:
    try:
        result = await runtime.run()
        entry = app[TASKS_KEY].get(runtime.run_id)
        if entry:
            entry["result"] = result
    except Exception as exc:
        entry = app[TASKS_KEY].get(runtime.run_id)
        if entry:
            entry["result"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}


async def _handle_agent_status(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    run_id = request.match_info["run_id"]
    entry = request.app[TASKS_KEY].get(run_id)
    if entry is None:
        return web.json_response({"error": "no such run"}, status=404)
    runtime = entry["runtime"]
    result = entry["result"]
    if result is not None:
        if isinstance(result, dict) and "error" in result:
            return web.json_response({"run_id": run_id, "error": result["error"]})
        return web.json_response(result.as_dict())
    return web.json_response(
        {
            "run_id": run_id,
            "goal": runtime.goal,
            "state": runtime.state.value,
            "steps_executed": runtime.steps_executed,
            "tool_calls": runtime.tool_calls,
            "failures": runtime.failures,
        }
    )


async def _handle_evidence(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    limit = int(request.query.get("limit", "50"))
    record_type = request.query.get("record_type")
    host_role = request.query.get("host_role")
    records = await services.evidence.query(
        record_type=EvidenceType(record_type) if record_type else None,
        host_role=host_role,
        limit=min(limit, 200),
    )
    return web.json_response({"count": len(records), "records": records})


async def _handle_introspect(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    report = await services.introspection.report()
    if services.foundation is not None:
        report["foundation"] = await services.foundation.audit()
    return web.json_response(report)


async def _handle_foundation(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    if services.foundation is None:
        return web.json_response({"error": "foundation audit unavailable"}, status=503)
    return web.json_response(await services.foundation.audit())


async def _handle_grounding(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    query = (request.query.get("q") or request.query.get("query") or "").strip()
    if not query:
        return web.json_response({"error": "missing q/query parameter"}, status=422)
    if services.grounding is None:
        return web.json_response({"error": "local grounding unavailable"}, status=503)
    return web.json_response(await services.grounding.ground(query))


async def _handle_hardware(request: web.Request) -> web.StreamResponse:
    denied = await _guard(request)
    if denied:
        return denied
    services = _services(request)
    snapshot = services.telemetry.snapshot()
    return web.json_response(snapshot.model_dump())
