from __future__ import annotations

import numpy as np

from myrag.config import EmbedConfig

_embedders: dict[tuple[str, str], "Embedder"] = {}


def _pick_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Embedder:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.cfg.model, device=self.device)
        return self._model

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        try:
            vecs = model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            if self.device != "cpu":
                self.device = "cpu"
                model = model.to("cpu")
                vecs = model.encode(
                    texts,
                    batch_size=self.cfg.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            else:
                raise
        return np.asarray(vecs, dtype=np.float32).tolist()

    def encode_query(self, text: str) -> list[float]:
        return self.encode_texts([text])[0]


def get_embedder(cfg: EmbedConfig) -> Embedder:
    key = (cfg.model, cfg.device)
    if key not in _embedders:
        _embedders[key] = Embedder(cfg)
    return _embedders[key]
