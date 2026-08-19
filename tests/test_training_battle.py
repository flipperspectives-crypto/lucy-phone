"""Adversarial / edge-case battle tests for the from-scratch training pipeline.

Stdlib-only. These stress the tokenizer, trainer, provider, standalone
``tiny_infer``, and introspection with malicious / degenerate / boundary inputs
to prove the sovereign pipeline fails closed and never lies about availability.

Slow, training-based cases are gated behind ``LUCY_SLOW=1`` so the default
suite stays fast (pure-Python training is intentionally slow by design).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import asyncio

from training.corpus import curate, ALLOWED_SOURCES, LICENSE_OWNED
from training.tokenizer import ByteTokenizer, BPETokenizer
from training.tiny_transformer import TinyTransformer
from training.lineage import LineageLedger, STATUS_DONE, STATUS_FAILED
from training.train import train
from training.provider import LocalLucyProvider
from lucy_edge.introspection.training_status import check_training

SLOW = os.environ.get("LUCY_SLOW") == "1"
_REPO = Path(__file__).resolve().parent.parent


def _random_ckpt(vocab=256, d=32, ctx=32, n_layers=1, ff_mult=4, seed=1) -> dict:
    m = TinyTransformer(vocab=vocab, d_model=d, ctx=ctx, n_layers=n_layers, ff_mult=ff_mult, seed=seed)
    return m.state_dict()


def _is_private_host(host: str) -> bool:
    host = host.split(":")[0]
    if host in ("localhost", "127.0.0.1"):
        return True
    parts = host.split(".")
    if len(parts) != 4:
        return False
    try:
        a, b, c, d = (int(p) for p in parts)
    except ValueError:
        return False
    return (
        a == 10
        or (a == 192 and b == 168)
        or (a == 172 and 16 <= b <= 31)
        or (a == 169 and b == 254)
    )


class TestSovereignty(unittest.TestCase):
    def test_training_imports_are_stdlib_only(self):
        """No network/cloud/ML package may enter the sovereign training path."""
        forbidden = {
            "socket", "requests", "httpx", "http", "grpc",
            "torch", "numpy", "transformers", "huggingface", "openai", "anthropic",
        }
        files = list((_REPO / "training").glob("*.py")) + [_REPO / "tiny_infer.py"]
        for f in files:
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        top = n.name.split(".")[0]
                        if top in forbidden:
                            self.fail(f"{f}: forbidden import {n.name}")
                        if top == "urllib" and n.name not in ("urllib.parse", "urllib.error"):
                            self.fail(f"{f}: urllib submodule {n.name} can dial out")
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    top = (node.module or "").split(".")[0]
                    if top in forbidden:
                        self.fail(f"{f}: forbidden import {node.module}")
                    if top == "urllib" and node.module not in ("urllib.parse", "urllib.error"):
                        self.fail(f"{f}: urllib submodule {node.module} can dial out")

    def test_corpus_ingests_no_external_urls(self):
        c = curate(".")
        urls = re.findall(r"https?://[^\s\"'<>]+", c.text)
        for u in urls:
            host = u.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0]
            self.assertTrue(_is_private_host(host), f"external URL in training corpus: {u}")
        for rec in c.manifest:
            self.assertEqual(rec.license, LICENSE_OWNED)


class TestTokenizerAdversarial(unittest.TestCase):
    def _bpe(self, target=512):
        tok = BPETokenizer(target_vocab=target)
        tok.train(["Lucy is local. Truth first.\n"])
        return tok

    def test_empty_round_trip(self):
        self.assertEqual(ByteTokenizer().decode(ByteTokenizer().encode("")), "")
        tok = self._bpe()
        self.assertEqual(tok.decode(tok.encode("")), "")

    def test_unicode_round_trip(self):
        tok = self._bpe()
        for s in ["Lauren \u2764\ufe0f \u65e5\u672c", "caf\u00e9 \u2615 \u4e2d\u6587", "\U0001F525\U0001F525\U0001F525"]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_single_char_repeat_bounded(self):
        tok = BPETokenizer(target_vocab=512)
        tok.train(["aaaaaaaaaaaa"])
        self.assertEqual(tok.decode(tok.encode("aaaa")), "aaaa")
        # Recursive BPE keeps merging (aa -> aaaa -> aaaaaaaa ...), so the vocab
        # grows past 257; the honest bound is the number of mergeable byte pairs
        # (<= bytes - 1) plus the 256 base ids.
        self.assertLessEqual(tok.vocab_size, 256 + 12)

    def test_oversized_target_vocab(self):
        tok = BPETokenizer(target_vocab=10_000_000)
        tok.train(["abc"])
        self.assertLessEqual(tok.vocab_size, 260)
        self.assertLessEqual(tok.vocab_size, 10_000_000)

    def test_whitespace_heavy_round_trip(self):
        tok = self._bpe()
        # BPE tokenizes on whitespace boundaries, so surrounding/redundant
        # whitespace collapses to single spaces between words -- the content is
        # preserved, which is the contract we assert here.
        self.assertEqual(tok.decode(tok.encode("a b c")), "a b c")
        self.assertEqual(tok.decode(tok.encode("  x  y  ")), "x y")


class TestProviderAdversarial(unittest.TestCase):
    def _ckpt(self, tmp, **kw):
        p = Path(tmp) / "ck.json"
        p.write_text(json.dumps(_random_ckpt(**kw)))
        return str(p)

    def test_empty_prompt_returns_empty(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._ckpt(tmp)
            prov = LocalLucyProvider(checkpoint_path=cp)
            r = asyncio.run(prov.generate("", model="lucy-local", max_new_tokens=8))
            self.assertEqual(r.text, "")
            self.assertEqual(r.eval_count, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_max_new_zero(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._ckpt(tmp)
            prov = LocalLucyProvider(checkpoint_path=cp)
            r = asyncio.run(prov.generate("hi", model="lucy-local", max_new_tokens=0))
            self.assertEqual(r.text, "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_malformed_checkpoint_unavailable(self):
        tmp = tempfile.mkdtemp()
        try:
            p = Path(tmp) / "bad.json"
            p.write_text("{not json")
            self.assertEqual(check_training(str(p), None)[0], "UNAVAILABLE")
            prov = LocalLucyProvider(checkpoint_path=str(p))
            with self.assertRaises(Exception):
                asyncio.run(prov.generate("x", model="lucy-local"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_tok_emb_unavailable(self):
        tmp = tempfile.mkdtemp()
        try:
            p = Path(tmp) / "ck.json"
            p.write_text(json.dumps({"vocab": 256, "d": 32, "ctx": 32, "L": 1, "ff": 128}))
            self.assertEqual(check_training(str(p), None)[0], "UNAVAILABLE")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unknown_tokenizer_type_falls_back(self):
        tmp = tempfile.mkdtemp()
        try:
            sd = _random_ckpt()
            sd["tokenizer"] = {"type": "future", "base": 256, "merges": [[65, 66]]}
            p = Path(tmp) / "ck.json"
            p.write_text(json.dumps(sd))
            prov = LocalLucyProvider(checkpoint_path=str(p))
            r = asyncio.run(prov.generate("ab", model="lucy-local", max_new_tokens=6))
            self.assertIsInstance(r.text, str)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(SLOW, "needs a trained checkpoint")
    def test_temperature_sampling_runs(self):
        from tests.helpers import trained_checkpoint

        tmp = tempfile.mkdtemp()
        try:
            cp = trained_checkpoint(tmp, steps=12)
            prov = LocalLucyProvider(checkpoint_path=cp)
            r = asyncio.run(prov.generate("Lauren", model="lucy-local", max_new_tokens=12, temperature=0.9))
            self.assertIsInstance(r.text, str)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTinyInferCLI(unittest.TestCase):
    def test_byte_level_live_checkpoint(self):
        r = subprocess.run(
            [sys.executable, "tiny_infer.py", "--checkpoint", "training/checkpoints/latest.json",
             "--prompt", "lauren", "--max-new", "8"],
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("generated", r.stdout)

    def test_missing_checkpoint_nonzero(self):
        r = subprocess.run(
            [sys.executable, "tiny_infer.py", "--checkpoint", "/no/such.json"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(r.returncode, 0)

    def test_dim_mismatch_nonzero(self):
        r = subprocess.run(
            [sys.executable, "tiny_infer.py", "--checkpoint", "training/checkpoints/latest.json",
             "--vocab", "999"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(r.returncode, 0)

    @unittest.skipUnless(SLOW, "trains a BPE checkpoint")
    def test_bpe_checkpoint(self):
        tmp = tempfile.mkdtemp()
        try:
            summary = train(
                repo_root=".", corpus_text="Lucy is local and loyal.\n",
                checkpoint_dir=Path(tmp) / "ck", steps=10, git_hash="t",
            )
            r = subprocess.run(
                [sys.executable, "tiny_infer.py", "--checkpoint", summary["latest"],
                 "--prompt", "lucy", "--max-new", "8"],
                capture_output=True, text=True, timeout=240,
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn("generated", r.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestIntrospectionAdversarial(unittest.TestCase):
    def _trained_ckpt(self, tmp, trained_steps=10, final_loss=2.0):
        sd = _random_ckpt()
        sd["trained_steps"] = trained_steps
        sd["final_loss"] = final_loss
        p = Path(tmp) / "ck.json"
        p.write_text(json.dumps(sd))
        return str(p)

    def test_running_row_hides_provenance(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._trained_ckpt(tmp)
            ledger_db = Path(tmp) / "lineage.db"
            ledger = LineageLedger(ledger_db)
            ledger.start_run(git_hash="abc", data_manifest_sha256="x", hyperparams={}, seed=1, checkpoint_path=cp)
            ledger.close()
            status, prov = check_training(cp, str(ledger_db))
            self.assertEqual(status, "AVAILABLE")
            self.assertNotIn("git_hash", prov)  # RUNNING is not DONE -> never claimed
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_id_absent_from_ledger(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._trained_ckpt(tmp)
            sd = json.loads(Path(cp).read_text())
            sd["lineage_run_id"] = "zzzmissing"
            Path(cp).write_text(json.dumps(sd))
            ledger_db = Path(tmp) / "lineage.db"
            LineageLedger(ledger_db).close()  # empty ledger
            status, prov = check_training(cp, str(ledger_db))
            self.assertEqual(status, "AVAILABLE")
            self.assertNotIn("git_hash", prov)
            # The phantom run_id must NOT leak into provenance: a checkpoint whose
            # embedded lineage_run_id has no matching ledger row resolves to honest
            # absence, never to a fabricated or stale attribution.
            self.assertNotIn("lineage_run_id", prov)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_two_rows_same_path_done_wins(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._trained_ckpt(tmp)
            ledger_db = Path(tmp) / "lineage.db"
            ledger = LineageLedger(ledger_db)
            r1 = ledger.start_run(git_hash="fail", data_manifest_sha256="x", hyperparams={}, seed=1, checkpoint_path=cp)
            ledger.finish_run(r1.run_id, STATUS_FAILED, final_loss=9.0)
            import time

            time.sleep(0.05)  # ensure the DONE row is unambiguously the latest
            r2 = ledger.start_run(git_hash="good", data_manifest_sha256="y", hyperparams={}, seed=1, checkpoint_path=cp)
            ledger.finish_run(r2.run_id, STATUS_DONE, final_loss=2.0)
            ledger.close()
            status, prov = check_training(cp, str(ledger_db))
            self.assertEqual(status, "AVAILABLE")
            self.assertEqual(prov["git_hash"], "good")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_directory_as_checkpoint(self):
        tmp = tempfile.mkdtemp()
        try:
            d = Path(tmp) / "dir"
            d.mkdir()
            self.assertEqual(check_training(str(d), None)[0], "UNAVAILABLE")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestIntegrityBoundary(unittest.TestCase):
    def _probe_ckpt(self, tmp):
        m = TinyTransformer(vocab=256, d_model=32, ctx=32)
        sd = m.state_dict()
        sd["trained_steps"] = 10
        sd["final_loss"] = 2.0
        seq = [1, 2, 3, 4]
        sd["probe_seq"] = seq
        logits, _ = m.forward([seq[:-1]])
        loss, _ = m.cross_entropy(logits, [seq[1:]])
        sd["probe_loss"] = loss
        p = Path(tmp) / "ck.json"
        p.write_text(json.dumps(sd))
        return str(p)

    def test_probe_loss_within_tolerance(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._probe_ckpt(tmp)
            prov = LocalLucyProvider(checkpoint_path=cp)
            self.assertTrue(prov.self_check())
            sd = json.loads(Path(cp).read_text())
            sd["probe_loss"] = sd["probe_loss"] + 0.4  # still within tolerance
            Path(cp).write_text(json.dumps(sd))
            self.assertTrue(prov.self_check())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_probe_loss_beyond_tolerance(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = self._probe_ckpt(tmp)
            sd = json.loads(Path(cp).read_text())
            sd["probe_loss"] = sd["probe_loss"] + 0.6  # outside tolerance -> tamper
            Path(cp).write_text(json.dumps(sd))
            prov = LocalLucyProvider(checkpoint_path=cp)
            self.assertFalse(prov.self_check())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_legacy_no_probe_trusted(self):
        tmp = tempfile.mkdtemp()
        try:
            p = Path(tmp) / "ck.json"
            p.write_text(json.dumps(_random_ckpt()))
            prov = LocalLucyProvider(checkpoint_path=str(p))
            self.assertTrue(prov.self_check())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_probe_seq_too_short(self):
        tmp = tempfile.mkdtemp()
        try:
            sd = _random_ckpt()
            sd["probe_seq"] = [1]
            sd["probe_loss"] = 1.0
            p = Path(tmp) / "ck.json"
            p.write_text(json.dumps(sd))
            prov = LocalLucyProvider(checkpoint_path=str(p))
            self.assertTrue(prov.self_check())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTrainingEdge(unittest.TestCase):
    @unittest.skipUnless(SLOW, "trains")
    def test_empty_corpus_raises(self):
        with self.assertRaises(RuntimeError):
            train(repo_root=".", corpus_text="", checkpoint_dir=tempfile.mkdtemp(), steps=5, git_hash="t")

    @unittest.skipUnless(SLOW, "trains")
    def test_corpus_shorter_than_ctx(self):
        tmp = tempfile.mkdtemp()
        try:
            s = train(repo_root=".", corpus_text="AB", checkpoint_dir=Path(tmp) / "ck",
                      steps=10, ctx=32, git_hash="t")
            self.assertTrue(math.isfinite(s["final_loss"]))
            self.assertTrue(Path(s["latest"]).exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(SLOW, "trains")
    def test_ctx_longer_than_corpus(self):
        tmp = tempfile.mkdtemp()
        try:
            s = train(repo_root=".", corpus_text="AB" * 5, checkpoint_dir=Path(tmp) / "ck",
                      steps=10, ctx=64, git_hash="t")
            self.assertTrue(math.isfinite(s["final_loss"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(SLOW, "trains")
    def test_high_lr_yields_finite_weights(self):
        tmp = tempfile.mkdtemp()
        try:
            s = train(repo_root=".", corpus_text="AB" * 20, checkpoint_dir=Path(tmp) / "ck",
                      steps=20, lr=50.0, git_hash="t")
            sd = json.loads(Path(s["latest"]).read_text())
            for k in ("tok_emb", "pos_emb"):
                for row in sd[k]:
                    for v in row:
                        self.assertTrue(math.isfinite(v))
            self.assertTrue(math.isfinite(s["final_loss"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(SLOW, "trains")
    def test_reproducible(self):
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            sa = train(repo_root=".", corpus_text="Lucy is local and loyal.\n",
                       checkpoint_dir=Path(a) / "ck", steps=20, seed=1, git_hash="t")
            sb = train(repo_root=".", corpus_text="Lucy is local and loyal.\n",
                       checkpoint_dir=Path(b) / "ck", steps=20, seed=1, git_hash="t")
            da = json.loads(Path(sa["latest"]).read_text())
            db = json.loads(Path(sb["latest"]).read_text())
            da.pop("lineage_run_id", None)
            db.pop("lineage_run_id", None)
            self.assertEqual(da, db)
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    @unittest.skipUnless(SLOW, "trains")
    def test_seed_sensitivity(self):
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            sa = train(repo_root=".", corpus_text="Lucy is local and loyal.\n",
                       checkpoint_dir=Path(a) / "ck", steps=20, seed=1, git_hash="t")
            sb = train(repo_root=".", corpus_text="Lucy is local and loyal.\n",
                       checkpoint_dir=Path(b) / "ck", steps=20, seed=2, git_hash="t")
            self.assertNotEqual(sa["final_loss"], sb["final_loss"])
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)


class TestCrossConsistency(unittest.TestCase):
    @unittest.skipUnless(SLOW, "trains")
    def test_training_and_tiny_infer_tokenizers_match(self):
        # Reconstructs the tokenizer the way standalone ``tiny_infer._load_tokenizer``
        # does (kept inline so the test never imports the self-contained script).
        tmp = tempfile.mkdtemp()
        try:
            s = train(repo_root=".", corpus_text="Lucy is local and loyal. Lauren is her source.\n",
                      checkpoint_dir=Path(tmp) / "ck", steps=10, git_hash="t")
            sd = json.loads(Path(s["latest"]).read_text())
            train_tok = BPETokenizer.from_state_dict(sd["tokenizer"])
            tok_state = sd.get("tokenizer")
            inf_tok = (
                BPETokenizer.from_state_dict(tok_state)
                if tok_state and tok_state.get("type") == "bpe"
                else ByteTokenizer()
            )
            sample = "Lucy serves Lauren with devotion and care."
            self.assertEqual(train_tok.encode(sample), inf_tok.encode(sample))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestKnownLimitation(unittest.TestCase):
    @unittest.skipUnless(SLOW, "trains")
    def test_greedy_collapse_is_deterministic_not_crash(self):
        from tests.helpers import trained_checkpoint

        tmp = tempfile.mkdtemp()
        try:
            cp = trained_checkpoint(tmp, steps=12)
            prov = LocalLucyProvider(checkpoint_path=cp)
            g1 = asyncio.run(prov.generate("Lauren", model="lucy-local", max_new_tokens=12, temperature=0.0)).text
            g2 = asyncio.run(prov.generate("Lauren", model="lucy-local", max_new_tokens=12, temperature=0.0)).text
            self.assertEqual(g1, g2)  # deterministic, even if degenerate
            self.assertIsInstance(g1, str)  # never crashes
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
