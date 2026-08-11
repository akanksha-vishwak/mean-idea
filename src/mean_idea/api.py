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
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding: ...

    def embedding_to_text(
        self,
        embedding: TextEmbedding,
        *,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> str: ...


def mean_embeddings(*embeddings: TextEmbedding) -> TextEmbedding:
    if not embeddings:
        raise ValueError("at least one embedding is required")

    shape = embeddings[0].values.shape
    if any(item.values.shape != shape for item in embeddings[1:]):
        raise ValueError("all embeddings must have the same shape")

    values = torch.stack([item.values.float() for item in embeddings]).mean(dim=0)
    return TextEmbedding(values)


def prepend_prompt(text: str, prompt: str | None = None) -> str:
    """Prepend an optional prompt to text before embedding it."""

    if prompt is not None and not isinstance(prompt, str):
        raise TypeError("prompt must be a string or None")
    return text if prompt is None else f"{prompt}{text}"


def interpolate_texts(
    model: LatentTextModel,
    *texts: str,
    prompt: str | None = None,
    canvas_length: int | None = None,
    multi_canvas: int | None = None,
) -> str:
    """Encode texts independently, average them, and decode the mean."""

    canvas_options = {}
    if canvas_length is not None:
        canvas_options["canvas_length"] = canvas_length
    if multi_canvas is not None:
        canvas_options["multi_canvas"] = multi_canvas
    embedding = mean_embeddings(
        *(
            model.text_to_embedding(
                prepend_prompt(text, prompt),
                **canvas_options,
            )
            for text in texts
        )
    )
    return model.embedding_to_text(embedding, **canvas_options)
