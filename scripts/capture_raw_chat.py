#!/usr/bin/env python3
"""Capture the real thesis Ollama streams and Modelfiles for diagnosis.

Run from the repository root:

    python scripts/capture_raw_chat.py

The default output prefix is ``docs/diagnostics/student-output-actual``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docs.diagnostics.capture_ollama_streams import main


if __name__ == "__main__":
    main()
