"""Tool permission tests: ALLOW / ASK / DENY and out-of-scope blocking."""

from __future__ import annotations

import unittest
from pathlib import Path

from lucy_edge.tools.permissions import (
    PermissionDecision,
    PermissionOutcome,
    PermissionPolicy,
    build_phone_policy,
)
from lucy_edge.tools.registry import ToolRegistry, ToolSpec

from .helpers import temp_dir


async def _noop_read(path="", **kw):
    return {"path": path, "content": "ok"}


class PermissionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.workspace = temp_dir()
        self.policy = build_phone_policy(self.workspace)
        Path(self.workspace, "a.txt").write_text("hello")
        Path(self.workspace, "operator.token").write_text("secret-value")

    def test_allow_read_within_scope(self):
        decision = self.policy.evaluate(
            "files.read_scoped", {"path": str(Path(self.workspace, "a.txt"))}
        )
        self.assertEqual(decision.outcome, PermissionOutcome.ALLOW)

    def test_ask_write_within_scope(self):
        decision = self.policy.evaluate(
            "files.write_scoped", {"path": str(Path(self.workspace, "b.txt"))}
        )
        self.assertEqual(decision.outcome, PermissionOutcome.ASK)

    def test_deny_out_of_scope_filesystem(self):
        outside = Path(temp_dir()) / "outside.txt"
        decision = self.policy.evaluate("files.read_scoped", {"path": str(outside)})
        self.assertEqual(decision.outcome, PermissionOutcome.DENY)

    def test_deny_arbitrary_shell(self):
        decision = self.policy.evaluate("shell.exec", {"command": "rm -rf /"})
        self.assertEqual(decision.outcome, PermissionOutcome.DENY)

    def test_deny_secret_path(self):
        decision = self.policy.evaluate(
            "files.read_scoped", {"path": str(Path(self.workspace, "operator.token"))}
        )
        self.assertEqual(decision.outcome, PermissionOutcome.DENY)

    def test_deny_destructive_git(self):
        decision = self.policy.evaluate("git.push", {})
        self.assertEqual(decision.outcome, PermissionOutcome.DENY)

    def test_allow_non_destructive_git(self):
        decision = self.policy.evaluate("git.status", {})
        self.assertEqual(decision.outcome, PermissionOutcome.ALLOW)

    def test_write_auto_allow_configuration(self):
        self.policy.write_auto_allow = True
        decision = self.policy.evaluate(
            "files.write_scoped", {"path": str(Path(self.workspace, "b.txt"))}
        )
        self.assertEqual(decision.outcome, PermissionOutcome.ALLOW)

    def test_delete_asks(self):
        decision = self.policy.evaluate(
            "files.delete_scoped", {"path": str(Path(self.workspace, "a.txt"))}
        )
        self.assertEqual(decision.outcome, PermissionOutcome.ASK)


class ToolRegistryPermissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_denies_shell_before_execution(self):
        policy = build_phone_policy(temp_dir())

        async def boom(**kw):
            raise AssertionError("shell tool must never execute")

        registry = ToolRegistry(policy)
        registry.register(ToolSpec("shell.exec", "arbitrary shell", boom, "shell"))
        result = await registry.execute("shell.exec", {"command": "id"}, context=None)
        self.assertTrue(result.denied)
        self.assertIn("denied", result.reason)

    async def test_registry_allows_scoped_read(self):
        workspace = temp_dir()
        target = Path(workspace, "a.txt")
        target.write_text("content")
        policy = build_phone_policy(workspace)
        registry = ToolRegistry(policy)
        registry.register(ToolSpec("files.read_scoped", "read", _noop_read, "read"))
        result = await registry.execute(
            "files.read_scoped", {"path": str(target)}, context=None
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.denied)
        self.assertIsNotNone(result.output_sha256)

    async def test_registry_blocks_out_of_scope(self):
        workspace = temp_dir()
        outside = Path(temp_dir(), "secret.txt")
        outside.write_text("secret")
        policy = build_phone_policy(workspace)
        registry = ToolRegistry(policy)
        registry.register(ToolSpec("files.read_scoped", "read", _noop_read, "read"))
        result = await registry.execute(
            "files.read_scoped", {"path": str(outside)}, context=None
        )
        self.assertTrue(result.denied)

    async def test_registry_blocks_secret_path(self):
        workspace = temp_dir()
        secret = Path(workspace, "operator.token")
        secret.write_text("top-secret")
        policy = build_phone_policy(workspace)
        registry = ToolRegistry(policy)
        registry.register(ToolSpec("files.read_scoped", "read", _noop_read, "read"))
        result = await registry.execute(
            "files.read_scoped", {"path": str(secret)}, context=None
        )
        self.assertTrue(result.denied)

    async def test_unknown_tool_is_denied(self):
        policy = build_phone_policy(temp_dir())
        registry = ToolRegistry(policy)
        result = await registry.execute("does.not.exist", {}, context=None)
        self.assertTrue(result.denied)


if __name__ == "__main__":
    unittest.main()
