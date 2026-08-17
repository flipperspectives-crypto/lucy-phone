"""From-scratch training pipeline tests: tokenizer, corpus, and gradient check.

These are stdlib-only and must run without torch/numpy or network access.
"""

from __future__ import annotations

import os
import unittest

from training.corpus import curate
from training.tokenizer import ByteTokenizer
from training.tiny_transformer import TinyTransformer
from training.lineage import LineageLedger, STATUS_DONE, STATUS_FAILED, STATUS_RUNNING
from training.train import train
from training.provider import LocalLucyProvider
from lucy_edge.introspection.training_status import check_training


class TestTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tok = ByteTokenizer()
        text = b"LUCY is local. Truth first."
        self.assertEqual(tok.decode(tok.encode(text)), text.decode())
        # vocab is exactly the 256 byte values
        self.assertEqual(tok.vocab_size, 256)


class TestCorpus(unittest.TestCase):
    def test_curate_is_provenance_tagged(self):
        corpus = curate(".")
        self.assertGreater(len(corpus.manifest), 0)
        for rec in corpus.manifest:
            self.assertTrue(rec.source)  # every record carries a source
            self.assertEqual(rec.license, "OWNED_LOCAL")
        manifest = corpus.manifest_dicts()
        self.assertIsInstance(manifest, list)
        self.assertGreater(len(manifest), 0)


class TestTinyTransformer(unittest.TestCase):
    def _batch(self, tok):
        return [tok.encode("LUCY is local.")[:8], tok.encode("Truth first.")[:8]]

    def test_grad_check_passes(self):
        """Regression test for the attention/FF backward bug (missing W1^T)."""
        tok = ByteTokenizer()
        m = TinyTransformer(vocab=256, d_model=32, ctx=32, n_layers=1, ff_mult=4, seed=1)
        batch = self._batch(tok)
        targets = [b[1:] + [0] for b in batch]
        max_err, passed = m.grad_check(batch, targets)
        self.assertTrue(passed, f"grad_check failed: max_err={max_err}")

    def test_grad_check_multi_layer(self):
        tok = ByteTokenizer()
        m = TinyTransformer(vocab=256, d_model=24, ctx=16, n_layers=2, ff_mult=4, seed=3)
        batch = [tok.encode("LUCY local truth.")[:6], tok.encode("Safe AI now.")[:6]]
        targets = [b[1:] + [0] for b in batch]
        max_err, passed = m.grad_check(batch, targets)
        self.assertTrue(passed, f"multi-layer grad_check failed: max_err={max_err}")

    def test_loss_decreases_under_sgd(self):
        tok = ByteTokenizer()
        m = TinyTransformer(vocab=256, d_model=32, ctx=32, n_layers=1, ff_mult=4, seed=1)
        batch = self._batch(tok)
        targets = [b[1:] + [0] for b in batch]
        lr = 0.05
        losses = []
        for _ in range(40):
            logits, cache = m.forward(batch)
            loss, dlogits = m.cross_entropy(logits, targets)
            losses.append(loss)
            m.backward(batch, logits, cache, dlogits)
            for name, g in m.grad.items():
                if isinstance(g[0], list):  # 2D param
                    for i in range(len(g)):
                        for j in range(len(g[i])):
                            m.params[name][i][j] -= lr * g[i][j]
                else:  # 1D param (layernorm gain/bias)
                    for i in range(len(g)):
                        m.params[name][i] -= lr * g[i]
        self.assertLess(losses[-1], losses[0])
        # meaningful learning, not just numerical noise
        self.assertLess(losses[-1], losses[0] * 0.6)


class TestLineage(unittest.TestCase):
    def test_run_lifecycle_is_recorded(self):
        import tempfile, os

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        try:
            ledger = LineageLedger(path)
            run = ledger.start_run(
                git_hash="abc123",
                data_manifest_sha256="deadbeef",
                hyperparams={"lr": 0.05, "d_model": 32},
                seed=1,
                checkpoint_path="ckpt_1.json",
            )
            self.assertEqual(run.status, STATUS_RUNNING)
            ledger.update(run.run_id, steps=10, final_loss=2.3)
            ledger.finish_run(run.run_id, STATUS_DONE, final_loss=1.1, note="ok")
            got = ledger.get(run.run_id)
            self.assertIsNotNone(got)
            self.assertEqual(got.status, STATUS_DONE)
            self.assertEqual(got.steps, 10)
            self.assertAlmostEqual(got.final_loss, 1.1)
            self.assertEqual(got.git_hash, "abc123")
            self.assertEqual(got.hyperparams["lr"], 0.05)
            self.assertEqual(len(ledger.all_runs()), 1)
            ledger.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_unknown_run_raises(self):
        import tempfile, os

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        try:
            ledger = LineageLedger(path)
            with self.assertRaises(KeyError):
                ledger.update("nope", steps=1)
            ledger.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestTrainAndProvider(unittest.TestCase):
    def _train_tmp(self):
        import tempfile, os

        tmp = tempfile.mkdtemp()
        ckpt = os.path.join(tmp, "ckpts")
        ledger_db = os.path.join(tmp, "lineage.db")
        summary = train(
            repo_root=".",
            checkpoint_dir=ckpt,
            steps=20,
            lr=0.05,
            ctx=32,
            d_model=32,
            n_layers=1,
            ff_mult=4,
            seed=1,
            batch_size=4,
            stride=8,
            lineage_db=ledger_db,
            git_hash="testhash",
        )
        return summary, ledger_db

    def test_train_produces_checkpoint_and_lineage(self):
        summary, ledger_db = self._train_tmp()
        self.assertTrue(os.path.exists(summary["latest"]))
        self.assertLess(summary["final_loss"], 6.0)  # learned something vs ~5.67 start
        ledger = LineageLedger(ledger_db)
        runs = ledger.all_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, STATUS_DONE)
        ledger.close()

    def test_provider_infers_from_checkpoint(self):
        import asyncio

        summary, _ = self._train_tmp()
        prov = LocalLucyProvider(checkpoint_path=summary["latest"], model_name="lucy-local")
        res = asyncio.run(prov.generate("LUCY is", model="lucy-local", max_new_tokens=8))
        self.assertEqual(res.provider, "local_lucy")
        self.assertFalse(res.simulated)
        self.assertIsInstance(res.text, str)

    def test_training_status_honest(self):
        import os

        summary, ledger_db = self._train_tmp()
        status, prov = check_training(summary["latest"], ledger_db)
        self.assertEqual(status, "AVAILABLE")
        self.assertEqual(prov["git_hash"], "testhash")
        # bogus path must stay UNAVAILABLE (never fabricated)
        self.assertEqual(check_training("/nonexistent.json", None)[0], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
