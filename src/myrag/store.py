from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from myrag.chunk import Chunk
from myrag.config import collection_name


def open_client(index_path: Path):
    index_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(index_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def open_collection(project_name: str, index_path: Path, embed_model: str | None = None):
    client = open_client(index_path)
    return client.get_or_create_collection(
        name=collection_name(project_name),
        metadata=_collection_metadata(embed_model),
    )


def reset_collection(project_name: str, index_path: Path, embed_model: str | None = None):
    client = open_client(index_path)
    name = collection_name(project_name)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, metadata=_collection_metadata(embed_model))


def _collection_metadata(embed_model: str | None) -> dict:
    meta = {"hnsw:space": "cosine"}
    if embed_model:
        meta["embed_model"] = embed_model
    return meta


def add_chunks(col, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    col.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text_with_prefix for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "doc_id": c.doc_id,
                "source_path": c.source_path,
                "title": c.title,
                "fmt": c.fmt,
                "section": c.section,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "text": c.text,
            }
            for c in chunks
        ],
    )


def index_stats(col) -> dict:
    got = col.get(include=["metadatas"])
    docs: dict[str, int] = {}
    for meta in got.get("metadatas") or []:
        doc_id = meta.get("doc_id", "?")
        docs[doc_id] = docs.get(doc_id, 0) + 1
    return {"chunks": col.count(), "documents": docs, "doc_chunks": docs}
