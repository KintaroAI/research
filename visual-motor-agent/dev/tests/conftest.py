"""Pytest setup: make ``dev/`` importable (``import world``) from any cwd."""

import sys
from pathlib import Path

DEV_DIR = Path(__file__).resolve().parent.parent
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))
