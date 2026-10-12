"""Chroma PersistentClient with telemetry disabled."""
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def persistent_client(path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
