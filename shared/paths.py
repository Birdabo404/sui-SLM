"""Shared default paths. One index name for the app, eval, query scripts, and KD."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_NAME = "lab-index"
DEFAULT_INDEX = os.environ.get("RAG_INDEX", str(ROOT / "rag" / INDEX_NAME))
