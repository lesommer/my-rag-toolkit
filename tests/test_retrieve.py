from dataclasses import dataclass, field

import pytest

from myrag.retrieve import retrieve
from myrag.store import add_chunks, open_collection


@dataclass
class FakeRerankCfg:
    enabled: bool = False
    model: str = ""


@dataclass
class FakeRetrieveCfg:
    top_k: int = 2
    candidate_k: int = 10
    score_threshold: float = 0.35
    min_candidates: int = 3
    rerank: FakeRerankCfg = field(default_factory=FakeRerankCfg)


class FakeEmbedder:
    def encode_query(self, text: str) -> list[float]:
        table = {
            "cat": [1.0, 0.0, 0.0],
            "dog": [0.0, 1.0, 0.0],
            "bird": [0.0, 0.0, 1.0],
            "feline pet": [0.9, 0.1, 0.0],
            "unrelated": [0.577, 0.577, 0.577],
        }
        return table[text]


def _chunk(doc_id, text):
    from myrag.chunk import Chunk

    return Chunk(
        chunk_id=f"{doc_id}__id",
        doc_id=doc_id,
        source_path=f"/tmp/{doc_id}",
        title=doc_id,
        fmt="md",
        section="",
        page=-1,
        chunk_index=0,
        token_count=10,
        text=text,
        text_with_prefix=f"Doc: {doc_id}\n---\n{text}",
    )


@pytest.fixture()
def col(tmp_path):
    collection = open_collection("t", tmp_path / "index", embed_model="fake")
    add_chunks(
        collection,
        [_chunk("cat.md", "about cats"), _chunk("dog.md", "about dogs"), _chunk("bird.md", "about birds")],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return collection


def test_retrieve_orders_by_similarity(col):
    contexts = retrieve("cat", col, FakeEmbedder(), FakeRetrieveCfg())
    assert contexts[0].doc_id == "cat.md"
    assert contexts[0].index == 1
    assert contexts[0].score > 0.9
    assert len(contexts) <= 2


def test_min_candidates_kept_below_threshold(col):
    cfg = FakeRetrieveCfg(score_threshold=0.99, min_candidates=3, top_k=3)
    contexts = retrieve("unrelated", col, FakeEmbedder(), cfg)
    assert len(contexts) == 3


def test_threshold_filters(col):
    cfg = FakeRetrieveCfg(score_threshold=0.8, min_candidates=1, top_k=3)
    contexts = retrieve("feline pet", col, FakeEmbedder(), cfg)
    assert contexts[0].doc_id == "cat.md"
    assert all(c.score >= 0.8 for c in contexts)


def test_retrieve_empty_index(tmp_path):
    collection = open_collection("empty", tmp_path / "index", embed_model="fake")
    with pytest.raises(RuntimeError):
        retrieve("cat", collection, FakeEmbedder(), FakeRetrieveCfg())
