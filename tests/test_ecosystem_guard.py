"""Enforce the ecosystem guard as a hard test gate.

If any production source introduces an outbound-capable import, a non-loopback
URL, or an export/external marker, this test fails -- making the sovereignty
staple impossible to regress silently.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


class EcosystemGuardTests(unittest.TestCase):
    def test_guard_passes(self):
        result = subprocess.run(
            [sys.executable, "scripts/ecosystem_guard.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"ecosystem_guard failed:\n{result.stdout}\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
