from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Context:
    index: int
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    section: str
    page: int
    text: str
    score: float
    token_count: int = 0


class Reranker:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            from sentence_transformers.cross_encoder import CrossEncoder
        from myrag.embed import _pick_device

        self._model = CrossEncoder(self.model_name, device=_pick_device(self.device))
        return self._model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        model = self._load()
        pairs = [[query, t] for t in texts]
        try:
            raw = model.predict(pairs)
        except Exception:
            model = model.to("cpu") if hasattr(model, "to") else model
            raw = model.predict(pairs)
        return [float(x) for x in np.asarray(raw).reshape(-1)]


def retrieve(query: str, col, embedder, rcfg, reranker: Reranker | None = None) -> list[Context]:
    total = col.count()
    if total == 0:
        raise RuntimeError("index is empty — run `rag build` first")
    query_vec = embedder.encode_query(query)
    k = max(1, min(rcfg.candidate_k, total))
    res = col.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]
    distances = res["distances"][0]
    cands = [
        {
            "id": cid,
            "meta": meta,
            "doc": document,
            "sim": 1.0 - float(dist),
        }
        for cid, meta, document, dist in zip(ids, metadatas, documents, distances)
    ]
    cands.sort(key=lambda c: c["sim"], reverse=True)

    kept = [c for c in cands if c["sim"] >= rcfg.score_threshold]
    if len(kept) < rcfg.min_candidates:
        kept = cands[: rcfg.min_candidates]

    if reranker is not None and kept:
        scores = reranker.scores(query, [c["doc"] for c in kept])
        for c, s in zip(kept, scores):
            c["rerank"] = s
        kept.sort(key=lambda c: c.get("rerank", 0.0), reverse=True)

    kept = kept[: rcfg.top_k]

    return [
        Context(
            index=i,
            chunk_id=c["id"],
            doc_id=c["meta"].get("doc_id", "?"),
            source_path=c["meta"].get("source_path", ""),
            title=c["meta"].get("title", ""),
            section=c["meta"].get("section", ""),
            page=int(c["meta"].get("page", -1)),
            text=c["meta"].get("text", c["doc"]),
            score=round(c["sim"], 4),
            token_count=int(c["meta"].get("token_count", 0)),
        )
        for i, c in enumerate(kept, start=1)
    ]
