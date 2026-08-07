from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class TextEmbedding:
    """A configured-length sequence of continuous token embeddings."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("embedding must have shape [sequence, hidden]")
        if not self.values.is_floating_point():
            raise TypeError("embedding values must be floating point")
        if not torch.isfinite(self.values).all():
            raise ValueError("embedding values must be finite")


class LatentTextModel(Protocol):
    """The model-agnostic two-call interface used by the experiment."""

    def text_to_embedding(self, text: str) -> TextEmbedding: ...

    def embedding_to_text(self, embedding: TextEmbedding) -> str: ...


def mean_embeddings(*embeddings: TextEmbedding) -> TextEmbedding:
    if not embeddings:
        raise ValueError("at least one embedding is required")

    shape = embeddings[0].values.shape
    if any(item.values.shape != shape for item in embeddings[1:]):
        raise ValueError("all embeddings must have the same shape")

    values = torch.stack([item.values.float() for item in embeddings]).mean(dim=0)
    return TextEmbedding(values)


def interpolate_texts(model: LatentTextModel, *texts: str) -> str:
    """Encode texts independently, average them, and decode the mean."""

    return model.embedding_to_text(
        mean_embeddings(*(model.text_to_embedding(text) for text in texts))
    )
