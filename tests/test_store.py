from myrag.chunk import Chunk
from myrag.config import collection_name
from myrag.store import add_chunks, index_stats, open_collection, reset_collection


def _chunk(doc_id, text, page=-1, section=""):
    digest = f"{doc_id}{text}"[: 12]
    return Chunk(
        chunk_id=f"{doc_id}__{abs(hash(text)) % 10**10:010d}",
        doc_id=doc_id,
        source_path=f"/tmp/{doc_id}",
        title=doc_id,
        fmt="md",
        section=section,
        page=page,
        chunk_index=0,
        token_count=10,
        text=text,
        text_with_prefix=f"Doc: {doc_id}\n---\n{text}",
    )


def test_collection_name_sanitization():
    assert collection_name("Eval Panel 2027") == "eval-panel-2027"
    assert collection_name("ab") == "rag-ab"
    assert collection_name("--x--") == "rx--0" or collection_name("--x--").startswith("r")
    n = collection_name("a" * 80)
    assert len(n) <= 63


def test_roundtrip_and_query(tmp_path):
    col = open_collection("proj", tmp_path / "index", embed_model="fake-model")
    chunks = [
        _chunk("a.md", "cat document", section="cats"),
        _chunk("b.md", "dog document", section="dogs"),
    ]
    add_chunks(col, chunks, embeddings=[[1.0, 0.0], [0.0, 1.0]])
    assert col.count() == 2

    res = col.query(query_embeddings=[[1.0, 0.0]], n_results=2, include=["metadatas", "distances"])
    assert res["ids"][0][0] == chunks[0].chunk_id
    meta = res["metadatas"][0][0]
    assert meta["doc_id"] == "a.md"
    assert meta["section"] == "cats"

    stats = index_stats(col)
    assert stats["chunks"] == 2
    assert stats["doc_chunks"] == {"a.md": 1, "b.md": 1}


def test_reset_collection(tmp_path):
    col = open_collection("proj", tmp_path / "index", embed_model="fake-model")
    add_chunks(col, [_chunk("a.md", "text")], embeddings=[[1.0, 0.0]])
    assert col.count() == 1
    col = reset_collection("proj", tmp_path / "index", embed_model="fake-model")
    assert col.count() == 0
    assert (col.metadata or {}).get("embed_model") == "fake-model"
