"""Optional frozen text encoder for speech embeddings."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class TextEncoder:
    """Thin wrapper around sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device

    @property
    def dim(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)
