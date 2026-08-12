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

    def text_to_embedding(
        self,
        text: str,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding: ...

    def embedding_to_text(
        self,
        embedding: TextEmbedding,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
        max_iterations: int | None = None,
    ) -> str: ...


def mean_embeddings(*embeddings: TextEmbedding) -> TextEmbedding:
    if not embeddings:
        raise ValueError("at least one embedding is required")

    shape = embeddings[0].values.shape
    if any(item.values.shape != shape for item in embeddings[1:]):
        raise ValueError("all embeddings must have the same shape")

    values = torch.stack([item.values.float() for item in embeddings]).mean(dim=0)
    return TextEmbedding(values)


def interpolate_texts(
    model: LatentTextModel,
    *texts: str,
    prompt: str | None = None,
    canvas_length: int | None = None,
    multi_canvas: int | None = None,
    max_iterations: int | None = None,
) -> str:
    """Encode texts independently, average them, and decode the mean."""

    embedding_options = {}
    if prompt is not None:
        embedding_options["prompt"] = prompt
    if canvas_length is not None:
        embedding_options["canvas_length"] = canvas_length
    if multi_canvas is not None:
        embedding_options["multi_canvas"] = multi_canvas
    embedding = mean_embeddings(
        *(
            model.text_to_embedding(
                text,
                **embedding_options,
            )
            for text in texts
        )
    )
    generation_options = dict(embedding_options)
    if max_iterations is not None:
        generation_options["max_iterations"] = max_iterations
    return model.embedding_to_text(embedding, **generation_options)
