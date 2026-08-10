#!/usr/bin/env python3
"""Nexus-Lucy Integration Bridge.

Bi-directional command relay between Lucy (local task dispatcher) and Nexus
(remote operator console).

    Lucy --status--> Bridge --status--> Nexus
    Nexus --SSE/WS--> Bridge --command--> Lucy

Features
    - Status reporter: forwards Lucy heartbeat + task state to Nexus
      ``/api/v1/node/status`` over HTTP POST.
    - Command listener: subscribes to Nexus ``/sse`` (with optional WebSocket
      fallback) and translates ``EXECUTE`` / ``PAUSE`` / ``ABORT`` directives
      into Lucy dispatcher signals.
    - Queue buffer: a SQLite-backed outbox (statuses destined for Nexus) and
      inbox (commands destined for Lucy) that survive Nexus/Lucy outages and
      flush on reconnect.
    - Health check: ``GET /health`` reporting bridge, Lucy and Nexus state.
    - Self-test: ``--self-test`` runs an offline, in-process integration test.

Run
    python3 bridge.py --config bridge.yaml
    python3 bridge.py --self-test

Endpoints exposed by the bridge
    POST /v1/lucy/status   receive status/heartbeat from Lucy, forward to Nexus
    GET  /health           bridge, Lucy and Nexus connectivity state
    GET  /                 bridge info
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import tempfile
import time
from contextlib import suppress
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

import aiohttp
import aiosqlite
import yaml
from aiohttp import web
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]
    HAS_WEBSOCKETS = False

__version__ = "0.1.0"

DEFAULT_CONFIG_PATH = "bridge.yaml"
MAX_BATCH = 100
MAX_BACKOFF_EXP = 6


class BridgeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "bridge.log"
    queue_db: str = "data/queue.db"
    flush_interval: float = 5.0
    reconnect_interval: float = 10.0
    heartbeat_interval: float = 0.0


class NexusConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str = "http://127.0.0.1:8080"
    status_path: str = "/api/v1/node/status"
    sse_path: str = "/sse"
    api_key: Optional[str] = None
    listener: str = "auto"
    fallback_after_failures: int = 3
    timeout: float = 10.0


class LucyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str = "lucy-node-01"
    status_path: str = "/v1/lucy/status"
    command_endpoint: str = "http://127.0.0.1:9700/v1/dispatch"
    timeout: float = 5.0
    stale_after: float = 60.0


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    nexus: NexusConfig = Field(default_factory=NexusConfig)
    lucy: LucyConfig = Field(default_factory=LucyConfig)


class TaskStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["heartbeat", "task"] = "task"
    task_id: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    node_id: Optional[str] = None
    ts: float = Field(default_factory=lambda: round(time.time(), 3))


class CommandEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(validation_alias=AliasChoices("command", "type"))
    task_id: str
    node_id: Optional[str] = None
    reason: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


def translate_command(cmd: CommandEvent) -> dict[str, Any]:
    """Translate a Nexus override directive into a Lucy dispatcher signal."""
    out: dict[str, Any] = {
        "action": cmd.command.lower(),
        "task_id": cmd.task_id,
        "source": "nexus",
        "ts": round(time.time(), 3),
    }
    if cmd.node_id:
        out["node_id"] = cmd.node_id
    if cmd.reason:
        out["reason"] = cmd.reason
    if cmd.payload:
        out["payload"] = cmd.payload
    return out


class Probe:
    """Cached connectivity state for an upstream/downstream service."""

    def __init__(self) -> None:
        self.ok = False
        self.last_ok: Optional[float] = None
        self.last_error: Optional[str] = None
        self.latency_ms: Optional[float] = None

    def success(self, latency_ms: float) -> None:
        self.ok = True
        self.last_ok = time.time()
        self.last_error = None
        self.latency_ms = round(latency_ms, 3)

    def failure(self, error: str) -> None:
        self.ok = False
        self.last_error = error

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "degraded",
            "last_success": _iso(self.last_ok),
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
        }


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class QueueStore:
    """SQLite-backed persistent queues (aiosqlite)."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._db: Optional[aiosqlite.Connection] = None
        self.outbox = Queue(self, "outbox")
        self.inbox = Queue(self, "inbox")

    async def open(self) -> None:
        parent = Path(self._path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS inbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        if self._db is None:
            raise RuntimeError("queue store is not open")
        cur = await self._db.execute(sql, params)
        await self._db.commit()
        return cur


class Queue:
    def __init__(self, store: "QueueStore", table: str) -> None:
        self._store = store
        self._table = table

    async def enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        await self._store.execute(
            f"INSERT INTO {self._table} (kind, payload, created_at, attempts) "
            "VALUES (?, ?, ?, 0)",
            (kind, json.dumps(payload), time.time()),
        )

    async def fetch_batch(self, limit: int = MAX_BATCH) -> list[dict[str, Any]]:
        cur = await self._store.execute(
            f"SELECT id, kind, payload, attempts FROM {self._table} "
            "ORDER BY id LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def delete(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ", ".join("?" * len(ids))
        await self._store.execute(
            f"DELETE FROM {self._table} WHERE id IN ({placeholders})", tuple(ids)
        )

    async def count(self) -> int:
        cur = await self._store.execute(
            f"SELECT COUNT(*) AS c FROM {self._table}"
        )
        row = await cur.fetchone()
        return int(row["c"]) if row else 0


class LucyNexusBridge:
    """Core bridge logic shared by the HTTP service and the self-test."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log = logging.getLogger("nexus_lucy_bridge")
        self._queue = QueueStore(config.bridge.queue_db)
        self._session: Optional[aiohttp.ClientSession] = None
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._started_at = time.monotonic()
        self._last_lucy_seen: Optional[float] = None
        self._probe_nexus = Probe()
        self._probe_lucy = Probe()

    async def start(self) -> None:
        await self._queue.open()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.nexus.timeout)
        )
        self._tasks = [
            asyncio.create_task(self._run_flusher()),
            asyncio.create_task(self._run_listener()),
            asyncio.create_task(self._run_heartbeat()),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self._queue.close()

    async def report_status(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Receive a status/heartbeat from Lucy, forward to Nexus or queue it."""
        try:
            status = TaskStatus.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"invalid status payload: {exc}") from exc

        payload = status.model_dump(exclude_none=True)
        payload.setdefault("node_id", self.config.lucy.node_id)
        self._last_lucy_seen = time.time()
        self._probe_lucy.success(0.0)

        forwarded = await self._try_post_nexus(payload)
        if not forwarded:
            await self._queue.outbox.enqueue("status", payload)
            self.log.warning("nexus unreachable, queued status task=%s", status.task_id)
        return {"queued": not forwarded, "forwarded": forwarded}

    async def health(self) -> dict[str, Any]:
        lucy_probe = self._probe_lucy
        age = None
        if self._last_lucy_seen is not None:
            age = round(time.time() - self._last_lucy_seen, 3)
        lucy_ok = (
            age is not None
            and age < self.config.lucy.stale_after
        ) or lucy_probe.ok
        lucy_status = "ok" if lucy_ok else ("unknown" if age is None else "stale")
        outbox_count = await self._queue.outbox.count()
        inbox_count = await self._queue.inbox.count()
        overall = "ok"
        if not (self._probe_nexus.ok or lucy_ok):
            overall = "degraded"
        return {
            "status": overall,
            "version": __version__,
            "timestamp": _iso(time.time()),
            "uptime_seconds": round(time.monotonic() - self._started_at, 3),
            "bridge": {"status": "ok"},
            "lucy": {
                "status": lucy_status,
                "last_seen": _iso(self._last_lucy_seen),
                "last_seen_age_seconds": age,
                "command_endpoint": self.config.lucy.command_endpoint,
                **lucy_probe.as_dict(),
            },
            "nexus": {
                **self._probe_nexus.as_dict(),
                "base_url": self.config.nexus.base_url,
                "queued": outbox_count,
            },
            "queue": {"outbox": outbox_count, "inbox": inbox_count},
        }

    async def _run_flusher(self) -> None:
        while not self._stop.is_set():
            try:
                await self._flush(self._queue.outbox, "status", self._try_post_nexus)
                await self._flush(self._queue.inbox, "command", self._try_dispatch_lucy)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.error("flush cycle failed: %s", exc)
            await self._wait_stop(self.config.bridge.flush_interval)

    async def _flush(
        self,
        queue: Queue,
        expected_kind: str,
        send: Any,
    ) -> None:
        rows = await queue.fetch_batch(MAX_BATCH)
        delivered: list[int] = []
        for row in rows:
            if row["kind"] != expected_kind:
                delivered.append(row["id"])
                continue
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                self.log.warning("dropping corrupt %s row %s", queue._table, row["id"])
                delivered.append(row["id"])
                continue
            if await send(payload):
                delivered.append(row["id"])
            else:
                break
        await queue.delete(delivered)

    async def _run_listener(self) -> None:
        mode = self.config.nexus.listener
        failures = 0
        while not self._stop.is_set():
            if mode == "ws" and HAS_WEBSOCKETS:
                connected = await self._ws_listen_loop_once()
            elif mode == "sse":
                connected = await self._sse_listen_loop_once()
            else:
                if HAS_WEBSOCKETS and failures >= self.config.nexus.fallback_after_failures:
                    connected = await self._ws_listen_loop_once()
                else:
                    connected = await self._sse_listen_loop_once()
            failures = 0 if connected else failures + 1
            delay = self._backoff(failures)
            if delay > 0:
                await self._wait_stop(delay)

    def _backoff(self, failures: int) -> float:
        exp = min(2 ** min(failures, MAX_BACKOFF_EXP), 64)
        return self.config.bridge.reconnect_interval * exp * random.uniform(0.8, 1.2)

    async def _sse_listen_loop_once(self) -> bool:
        url = self._nexus_url(self.config.nexus.sse_path)
        connected = False
        try:
            timeout = aiohttp.ClientTimeout(
                total=None, sock_connect=self.config.nexus.timeout
            )
            async with self._session.get(
                url, headers=self._nexus_headers(), timeout=timeout
            ) as resp:
                if resp.status != 200:
                    raise ConnectionError(f"SSE http {resp.status}")
                connected = True
                self._probe_nexus.success(0.0)
                self.log.info("connected to nexus SSE: %s", url)
                async for event in self._iter_sse(resp):
                    await self._handle_command_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if connected:
                self.log.warning("sse stream ended: %s", exc)
            else:
                self.log.warning("sse connect failed (%s): %s", url, exc)
        return connected

    async def _ws_listen_loop_once(self) -> bool:
        if not HAS_WEBSOCKETS:
            return False
        url = self._nexus_ws_url()
        connected = False
        try:
            kwargs: dict[str, Any] = {"ping_interval": 20, "ping_timeout": 20}
            headers: dict[str, str] = {}
            if self.config.nexus.api_key:
                headers["Authorization"] = f"Bearer {self.config.nexus.api_key}"
            try:
                kwargs["additional_headers"] = headers
                conn = websockets.connect(url, **kwargs)
            except TypeError:  # older websockets versions
                kwargs.pop("additional_headers", None)
                kwargs["extra_headers"] = headers
                conn = websockets.connect(url, **kwargs)
            async with conn as socket:
                connected = True
                self._probe_nexus.success(0.0)
                self.log.info("connected to nexus WS: %s", url)
                async for message in socket:
                    if isinstance(message, bytes):
                        message = message.decode("utf-8", "replace")
                    try:
                        raw = json.loads(message)
                    except json.JSONDecodeError:
                        self.log.warning("ignoring non-json ws message")
                        continue
                    items = raw if isinstance(raw, list) else [raw]
                    for item in items:
                        await self._handle_command(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if connected:
                self.log.warning("ws stream ended: %s", exc)
            else:
                self.log.warning("ws connect failed (%s): %s", url, exc)
        return connected

    async def _iter_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[dict[str, str]]:
        buf = b""
        event_type = "message"
        data_lines: list[str] = []
        async for chunk in resp.content.iter_any():
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_text = line.decode("utf-8", "replace").rstrip("\r")
                if line_text == "":
                    if data_lines:
                        yield {"event": event_type, "data": "\n".join(data_lines)}
                    event_type = "message"
                    data_lines = []
                elif line_text.startswith("event:"):
                    event_type = line_text[len("event:"):].strip() or event_type
                elif line_text.startswith("data:"):
                    data_lines.append(line_text[len("data:"):].strip())
                elif not line_text.startswith(":"):
                    data_lines.append(line_text)
        if data_lines:
            yield {"event": event_type, "data": "\n".join(data_lines)}

    async def _handle_command_event(self, event: dict[str, str]) -> None:
        if event["event"] not in ("command", "directive", "message"):
            return
        try:
            raw = json.loads(event["data"])
        except json.JSONDecodeError:
            self.log.warning("ignoring non-json sse data")
            return
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            await self._handle_command(item)

    async def _handle_command(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        try:
            cmd = CommandEvent.model_validate(raw)
        except ValidationError as exc:
            self.log.warning("invalid command event: %s", exc)
            return
        if cmd.node_id and cmd.node_id != self.config.lucy.node_id:
            self.log.debug("command %s not addressed to %s", cmd.task_id, self.config.lucy.node_id)
            return
        payload = translate_command(cmd)
        delivered = await self._try_dispatch_lucy(payload)
        if not delivered:
            await self._queue.inbox.enqueue("command", payload)
            self.log.warning("lucy unreachable, queued command %s task=%s", cmd.command, cmd.task_id)
        self.log.info(
            "command %s task=%s delivered=%s", cmd.command, cmd.task_id, delivered
        )

    async def _run_heartbeat(self) -> None:
        interval = self.config.bridge.heartbeat_interval
        if interval <= 0:
            return
        while not self._stop.is_set():
            await self._wait_stop(interval)
            if self._stop.is_set():
                break
            payload = {
                "type": "heartbeat",
                "node_id": self.config.lucy.node_id,
                "status": "online",
                "bridge_version": __version__,
                "ts": round(time.time(), 3),
            }
            if not await self._try_post_nexus(payload):
                self.log.debug("bridge heartbeat not delivered to nexus")

    async def _try_post_nexus(self, payload: dict[str, Any]) -> bool:
        url = self._nexus_url(self.config.nexus.status_path)
        try:
            t0 = time.monotonic()
            async with self._session.post(
                url, json=payload, headers=self._nexus_headers()
            ) as resp:
                latency = (time.monotonic() - t0) * 1000
                if resp.status < 400:
                    self._probe_nexus.success(latency)
                    return True
                self._probe_nexus.failure(f"http {resp.status}")
                return False
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            self._probe_nexus.failure(f"{type(exc).__name__}: {exc}")
            return False

    async def _try_dispatch_lucy(self, payload: dict[str, Any]) -> bool:
        url = self.config.lucy.command_endpoint
        try:
            t0 = time.monotonic()
            async with self._session.post(url, json=payload) as resp:
                latency = (time.monotonic() - t0) * 1000
                if resp.status < 400:
                    self._probe_lucy.success(latency)
                    return True
                self._probe_lucy.failure(f"http {resp.status}")
                return False
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            self._probe_lucy.failure(f"{type(exc).__name__}: {exc}")
            return False

    def _nexus_url(self, path: str) -> str:
        return f"{self.config.nexus.base_url.rstrip('/')}{path}"

    def _nexus_ws_url(self) -> str:
        return self._nexus_url(self.config.nexus.sse_path).replace(
            "http://", "ws://", 1
        ).replace("https://", "wss://", 1)

    def _nexus_headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self.config.nexus.api_key:
            headers["Authorization"] = f"Bearer {self.config.nexus.api_key}"
        return headers

    async def _wait_stop(self, delay: float) -> None:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=max(delay, 0.0))


def create_app(bridge: LucyNexusBridge) -> web.Application:
    app = web.Application()
    app["bridge"] = bridge
    app.router.add_post("/v1/lucy/status", _handle_lucy_status)
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/", _handle_index)
    return app


async def _handle_lucy_status(request: web.Request) -> web.StreamResponse:
    bridge = request.app["bridge"]
    try:
        raw = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(raw, dict):
        return web.json_response({"error": "expected a JSON object"}, status=400)
    try:
        result = await bridge.report_status(raw)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=422)
    return web.json_response(result, status=202 if result["queued"] else 200)


async def _handle_health(request: web.Request) -> web.StreamResponse:
    bridge = request.app["bridge"]
    return web.json_response(await bridge.health())


async def _handle_index(request: web.Request) -> web.StreamResponse:
    bridge = request.app["bridge"]
    return web.json_response(
        {
            "service": "nexus-lucy-bridge",
            "version": __version__,
            "node_id": bridge.config.lucy.node_id,
            "endpoints": {
                "lucy_status": bridge.config.lucy.status_path,
                "health": "/health",
            },
        }
    )


def setup_logging(level: str, log_dir: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger("nexus_lucy_bridge")
    logger.setLevel(level.upper())
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            Path(log_dir) / log_file,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


_ENV_OVERRIDES: dict[str, tuple[str, str, str, Any]] = {
    "bridge.port": ("bridge", "port", "BRIDGE_PORT", int),
    "bridge.host": ("bridge", "host", "BRIDGE_HOST", str),
    "bridge.log_level": ("bridge", "log_level", "BRIDGE_LOG_LEVEL", str),
    "bridge.queue_db": ("bridge", "queue_db", "BRIDGE_QUEUE_DB", str),
    "nexus.base_url": ("nexus", "base_url", "NEXUS_BASE_URL", str),
    "nexus.api_key": ("nexus", "api_key", "NEXUS_API_KEY", str),
    "nexus.sse_path": ("nexus", "sse_path", "NEXUS_SSE_PATH", str),
    "nexus.status_path": ("nexus", "status_path", "NEXUS_STATUS_PATH", str),
    "lucy.node_id": ("lucy", "node_id", "LUCY_NODE_ID", str),
    "lucy.command_endpoint": ("lucy", "command_endpoint", "LUCY_COMMAND_ENDPOINT", str),
    "lucy.status_path": ("lucy", "status_path", "LUCY_STATUS_PATH", str),
}


def load_config(path: str) -> Config:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config file not found: {cfg_path} (create bridge.yaml or pass --config)"
        )
    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")
    config = Config.model_validate(data)
    return _apply_env_overrides(config)


def _apply_env_overrides(config: Config) -> Config:
    data = config.model_dump()
    for env_name, (section, key, env_var, cast) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        try:
            data[section][key] = cast(value)
        except ValueError:
            logging.getLogger("nexus_lucy_bridge").warning(
                "invalid %s value for %s", env_var, env_name
            )
    return Config.model_validate(data)


async def run_bridge(config: Config) -> int:
    log = logging.getLogger("nexus_lucy_bridge")
    bridge = LucyNexusBridge(config)
    await bridge.start()
    app = create_app(bridge)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.bridge.host, config.bridge.port)
    await site.start()
    log.info(
        "bridge listening on http://%s:%s node=%s nexus=%s",
        config.bridge.host,
        config.bridge.port,
        config.lucy.node_id,
        config.nexus.base_url,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await site.stop()
        await runner.cleanup()
        await bridge.stop()
    log.info("bridge stopped")
    return 0


async def run_self_test(config_path: str) -> int:
    print("== nexus_lucy_bridge self-test ==")
    logging.basicConfig(level=logging.ERROR)
    failures: list[str] = []

    async def check(name: str, coro: Any) -> None:
        try:
            await coro
            print(f"[PASS] {name}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] {name}: {exc}")

    if config_path and Path(config_path).exists():
        async def _cfg() -> None:
            cfg = load_config(config_path)
            assert cfg.lucy.node_id
        await check(f"config loads from {config_path}", _cfg())
    else:
        print(f"[SKIP] config load ({config_path} not present)")

    async def _translate() -> None:
        cmd = CommandEvent.model_validate(
            {"type": "PAUSE", "task_id": "t1", "reason": "operator override"}
        )
        payload = translate_command(cmd)
        assert payload["action"] == "pause"
        assert payload["task_id"] == "t1"
        assert payload["source"] == "nexus"
    await check("translate_command maps directives", _translate())

    async def _integration() -> None:
        state: dict[str, Any] = {
            "status_posts": [],
            "commands": [],
            "offline": False,
            "lucy_offline": False,
            "sse_queue": asyncio.Queue(),
            "sse_connected": asyncio.Event(),
        }

        async def nexus_status(request: web.Request) -> web.StreamResponse:
            if state["offline"]:
                return web.Response(status=503, text="offline")
            state["status_posts"].append(await request.json())
            return web.Response(status=204)

        async def nexus_sse(request: web.Request) -> web.StreamResponse:
            if state["offline"]:
                return web.Response(status=503, text="offline")
            resp = web.StreamResponse(
                status=200,
                reason="OK",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            await resp.prepare(request)
            state["sse_connected"].set()
            try:
                while True:
                    event = await asyncio.wait_for(
                        state["sse_queue"].get(), timeout=10
                    )
                    if event is None:
                        break
                    await resp.write(
                        f"event: command\ndata: {json.dumps(event)}\n\n".encode()
                    )
            except (ConnectionResetError, asyncio.CancelledError, asyncio.TimeoutError):
                pass
            with suppress(ConnectionResetError):
                await resp.write_eof()
            return resp

        async def lucy_dispatch(request: web.Request) -> web.StreamResponse:
            if state["lucy_offline"]:
                return web.Response(status=503, text="offline")
            state["commands"].append(await request.json())
            return web.json_response({"ok": True})

        nexus_app = web.Application()
        nexus_app.router.add_post("/api/v1/node/status", nexus_status)
        nexus_app.router.add_get("/sse", nexus_sse)
        lucy_app = web.Application()
        lucy_app.router.add_post("/v1/dispatch", lucy_dispatch)

        nexus_runner = web.AppRunner(nexus_app)
        lucy_runner = web.AppRunner(lucy_app)
        await nexus_runner.setup()
        await lucy_runner.setup()
        nexus_site = web.TCPSite(nexus_runner, "127.0.0.1", 0)
        lucy_site = web.TCPSite(lucy_runner, "127.0.0.1", 0)
        await nexus_site.start()
        await lucy_site.start()
        nexus_port = nexus_runner.addresses[0][1]
        lucy_port = lucy_runner.addresses[0][1]

        tmp_db = tempfile.mktemp(prefix="bridge_selftest_", suffix=".db")
        try:
            config = Config(
                bridge=BridgeConfig(
                    queue_db=tmp_db,
                    flush_interval=0.2,
                    reconnect_interval=0.3,
                    heartbeat_interval=0.0,
                ),
                nexus=NexusConfig(base_url=f"http://127.0.0.1:{nexus_port}"),
                lucy=LucyConfig(
                    node_id="lucy-selftest",
                    command_endpoint=f"http://127.0.0.1:{lucy_port}/v1/dispatch",
                ),
            )
            bridge = LucyNexusBridge(config)
            await bridge.start()

            async def wait_until(cond: Any, what: str) -> None:
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if cond():
                        return
                    await asyncio.sleep(0.05)
                raise AssertionError(f"timed out waiting for {what}")

            result = await bridge.report_status(
                {"type": "task", "task_id": "t1", "status": "running"}
            )
            assert result["forwarded"], result
            assert state["status_posts"] and state["status_posts"][0]["task_id"] == "t1"

            await asyncio.wait_for(state["sse_connected"].wait(), 5.0)
            await state["sse_queue"].put(
                {"command": "EXECUTE", "task_id": "t9", "node_id": "lucy-selftest"}
            )
            await wait_until(
                lambda: any(c["task_id"] == "t9" for c in state["commands"]),
                "command t9 to reach lucy",
            )

            state["lucy_offline"] = True
            await state["sse_queue"].put({"type": "PAUSE", "task_id": "t10"})
            await wait_until(
                lambda: asyncio.get_event_loop() is not None and True,
                "probe",
            )
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if await bridge._queue.inbox.count() >= 1:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("timed out waiting for queued command t10")

            state["lucy_offline"] = False
            await wait_until(
                lambda: any(c["task_id"] == "t10" for c in state["commands"]),
                "queued command t10 to flush to lucy",
            )

            state["offline"] = True
            result = await bridge.report_status(
                {"type": "task", "task_id": "t2", "status": "pending"}
            )
            assert result["queued"], result
            health = await bridge.health()
            assert health["nexus"]["queued"] >= 1
            assert health["nexus"]["status"] == "degraded"

            state["offline"] = False
            await wait_until(
                lambda: any(p.get("task_id") == "t2" for p in state["status_posts"]),
                "queued status t2 to flush to nexus",
            )
            health = await bridge.health()
            assert health["nexus"]["queued"] == 0
            assert await bridge._queue.outbox.count() == 0

            raised = False
            try:
                await bridge.report_status({"type": "bogus"})
            except ValueError:
                raised = True
            assert raised, "invalid status payload must be rejected"
        finally:
            await bridge.stop()
            with suppress(Exception):
                state["sse_queue"].put_nowait(None)
            await asyncio.sleep(0.1)
            with suppress(Exception):
                Path(tmp_db).unlink()
            await lucy_runner.cleanup()
            await nexus_runner.cleanup()

    await check("status forward / queue flush / command relay / health", _integration())

    print("== self-test complete ==")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bridge.py",
        description="Nexus-Lucy Integration Bridge",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="path to bridge.yaml config"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the offline self-test and exit",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(f"nexus-lucy-bridge {__version__}")
        return 0

    if args.self_test:
        return asyncio.run(run_self_test(args.config))

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    setup_logging(
        config.bridge.log_level, config.bridge.log_dir, config.bridge.log_file
    )
    return asyncio.run(run_bridge(config))


if __name__ == "__main__":
    sys.exit(main())
