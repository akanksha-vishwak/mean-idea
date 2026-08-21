from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


@dataclass(frozen=True, slots=True)
class TextEmbedding:
    """A configured-length sequence of continuous token embeddings."""

    values: torch.Tensor
    noise: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("embedding must have shape [sequence, hidden]")
        if not self.values.is_floating_point():
            raise TypeError("embedding values must be floating point")
        if not torch.isfinite(self.values).all():
            raise ValueError("embedding values must be finite")
        if self.noise is not None:
            if self.noise.shape != self.values.shape[:1]:
                raise ValueError("noise must have shape [sequence]")
            if not self.noise.is_floating_point():
                raise TypeError("noise values must be floating point")
            if not torch.isfinite(self.noise).all() or (self.noise < 0).any():
                raise ValueError("noise values must be finite and non-negative")


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

    values = torch.stack([item.values.float() for item in embeddings])
    noise = torch.stack(
        [
            item.noise.float()
            if item.noise is not None
            else torch.zeros(shape[0], device=item.values.device)
            for item in embeddings
        ]
    )
    weights = torch.softmax(-noise, dim=0)
    combined_values = (values * weights.unsqueeze(-1)).sum(dim=0)
    combined_noise = (noise * weights).sum(dim=0)
    return TextEmbedding(combined_values, combined_noise)


def interpolate_texts(
    model: LatentTextModel,
    *texts: str,
    prompts: Sequence[str | None] | None = None,
    canvas_length: int | None = None,
    multi_canvas: int | None = None,
    max_iterations: int | None = None,
) -> str:
    """Encode texts independently, average them, and decode the mean."""

    if isinstance(prompts, (str, bytes)):
        raise ValueError("prompts must be a sequence of strings or None")
    source_prompts = (None,) * len(texts) if prompts is None else tuple(prompts)
    if len(source_prompts) != len(texts):
        raise ValueError("prompts must have the same length as texts")
    if any(
        prompt is not None and not isinstance(prompt, str)
        for prompt in source_prompts
    ):
        raise ValueError("prompts must contain only strings or None")

    embedding_options = {}
    if canvas_length is not None:
        embedding_options["canvas_length"] = canvas_length
    if multi_canvas is not None:
        embedding_options["multi_canvas"] = multi_canvas
    embedding = mean_embeddings(
        *(
            model.text_to_embedding(
                text,
                prompt=prompt,
                **embedding_options,
            )
            for text, prompt in zip(texts, source_prompts, strict=True)
        )
    )
    generation_options = dict(embedding_options)
    if max_iterations is not None:
        generation_options["max_iterations"] = max_iterations
    return model.embedding_to_text(embedding, **generation_options)
