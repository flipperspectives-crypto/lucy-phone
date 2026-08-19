"""Sovereignty contract: nothing external enters, everything stays loopback.

This is the runtime companion to scripts/ecosystem_guard.py.  It asserts the
non-negotiable invariants of the Lucy ecosystem:

  * the phone client and bridge only ever address loopback addresses,
  * a cloud / non-loopback endpoint is classified as PUBLIC_CLOUD (rejected),
  * training data comes exclusively from files already inside this repo
    (no remote fetch, no external corpus).
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from lucy_edge.foundation.audit import (
    ENDPOINT_LOCAL_LOOPBACK,
    ENDPOINT_PUBLIC_CLOUD,
    classify_endpoint,
)
from lucy_edge.phone.client import LucyEdgeClient
from training.corpus import curate

REPO_ROOT = Path(__file__).parent.parent


def _is_loopback(url: str) -> bool:
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    return host in ("127.0.0.1", "localhost", "::1")


class SovereigntyContractTests(unittest.TestCase):
    def test_phone_client_default_is_loopback(self):
        default = inspect.signature(LucyEdgeClient.__init__).parameters["base_url"].default
        self.assertTrue(_is_loopback(default), f"client base_url not loopback: {default!r}")

    def test_bridge_defaults_are_loopback(self):
        # Imported lazily to avoid dragging the bridge's network deps into every run.
        from bridge import Config

        cfg = Config()
        self.assertEqual(cfg.bridge.host, "127.0.0.1")
        self.assertTrue(_is_loopback(cfg.nexus.base_url), cfg.nexus.base_url)
        self.assertTrue(_is_loopback(cfg.lucy.command_endpoint), cfg.lucy.command_endpoint)

    def test_cloud_endpoint_is_rejected(self):
        self.assertEqual(classify_endpoint("https://api.openai.com/v1"), ENDPOINT_PUBLIC_CLOUD)
        self.assertEqual(classify_endpoint("http://ollama.lan:11434"), ENDPOINT_PUBLIC_CLOUD)
        self.assertEqual(classify_endpoint("http://127.0.0.1:8970"), ENDPOINT_LOCAL_LOOPBACK)

    def test_training_corpus_is_local_only(self):
        corpus = curate(str(REPO_ROOT))
        self.assertGreater(len(corpus.manifest), 0)
        for record in corpus.manifest:
            self.assertNotIn("://", record.source, f"remote source in corpus: {record.source}")
            if record.source.startswith("synthesized:"):
                continue
            # Every owned source text lives inside this repo.
            self.assertTrue(
                (REPO_ROOT / record.source).exists(),
                f"corpus source not found in repo: {record.source}",
            )

    def test_remote_checkpoint_is_not_fetched(self):
        """A tampered config pointing at a URL cannot pull external data.

        Even if an operator sets training.checkpoint_path to a remote URL, the
        provider must never fetch it -- it stays unavailable.  This is the
        backstop that makes 'nothing from the outside comes inside' foolproof
        against config tampering.
        """
        from training.provider import LocalLucyProvider

        prov = LocalLucyProvider(checkpoint_path="https://example.com/evil/model.json")
        # No network call: the path simply does not exist on the local filesystem.
        self.assertFalse(prov.self_check())


if __name__ == "__main__":
    unittest.main()
