"""Command-line launcher for the Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    if not app_path.is_file():
        raise SystemExit(f"Cannot find the Streamlit application at {app_path}.")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
