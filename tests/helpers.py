"""Shared test helpers.

Tests use temp directories only.  No live model, no remote inference host, no
network to any real inference endpoint.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from lucy_edge.config import LucyEdgeConfig
from lucy_edge.services import build_services


class FakeTransport:
    """Deterministic fake HTTP transport for the Ollama provider.

    Records every request (method, path, payload) and returns canned data,
    so provider client logic can be tested with zero network activity.
    """

    def __init__(
        self,
        responses: Optional[dict[tuple[str, str], Any]] = None,
        fail_with: Optional[Exception] = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_with = fail_with
        self.calls: list[tuple[str, str, Optional[dict]]] = []

    def on(self, method: str, path: str, data: Any) -> "FakeTransport":
        self.responses[(method, path)] = data
        return self

    async def request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        self.calls.append((method, path, payload))
        if self.fail_with is not None:
            raise self.fail_with
        key = (method, path)
        if key not in self.responses:
            raise AssertionError(f"no canned response for {key}")
        return self.responses[key]


def make_config(
    tmp: str,
    host_role: str = "PHONE",
    host_id: str = "test-host",
    phone_local_inference: bool = False,
    phone_local_inference_unlocked: bool | None = None,
) -> LucyEdgeConfig:
    config = LucyEdgeConfig(base_dir=tmp, host_role=host_role, host_id=host_id)
    config.phone.phone_local_inference_enabled = phone_local_inference
    # Default unlocked to match enabled so routing-logic tests focus on
    # routing, not the ARM guard. ARM-guard tests pass unlocked=False
    # explicitly to exercise the fail-closed path.
    config.phone.local_inference_unlocked = (
        phone_local_inference
        if phone_local_inference_unlocked is None
        else phone_local_inference_unlocked
    )
    config.memory.db_path = f"{tmp}/memory.db"
    config.evidence.dir_path = f"{tmp}/evidence"
    config.evidence.ledger_db = f"{tmp}/evidence.db"
    config.phone_client.token_file = f"{tmp}/operator.token"
    return config


def temp_dir() -> str:
    return tempfile.mkdtemp(prefix="lucy_edge_test_")


def local_checkpoint(tmp: str) -> str:
    """Write a real (tiny, randomly-initialized) ``local_lucy`` checkpoint so
    ``LocalLucyProvider`` registers, reports healthy, AND can actually run
    on-device generation in tests.

    The weights are random (not trained), so generated text is meaningless --
    the honest "model_weights_present" GAP is about *trained* quality, not file
    presence.  Real generation requires a trained checkpoint (Step 2).
    """
    cp = Path(tmp) / "lucy_local_checkpoint.json"
    from training.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab=256, d_model=64, ctx=32, n_layers=2, ff_mult=4, seed=1)
    sd = model.state_dict()
    sd.update({"vocab": 256, "d": 64, "ctx": 32, "L": 2, "ff": 4, "trained_steps": 5, "final_loss": 3.21})
    cp.write_text(json.dumps(sd))
    return str(cp)


def make_services(tmp: str, **kwargs: Any):
    config = make_config(tmp, **kwargs)
    return build_services(config, transport=None, fixed_token="test-token")


async def wait_until(cond, what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await __import__("asyncio").sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")
