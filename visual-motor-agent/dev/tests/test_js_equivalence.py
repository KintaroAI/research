"""The browser renderer/simulator (circle_world.js) matches the Python world.

Runs ``python -m world.check_js_equivalence`` (about 300 fixtures: pixel-exact
renders, step outputs within 1e-5).  Skipped when node is not installed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEV_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_matches_python():
    proc = subprocess.run([sys.executable, "-m", "world.check_js_equivalence"], cwd=DEV_DIR,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
