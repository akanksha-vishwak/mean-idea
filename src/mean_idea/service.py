from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from mean_idea.api import (
    LatentTextModel,
    TextEmbedding,
    interpolate_texts,
    prepend_prompt,
)
from mean_idea.remote import PROTOCOL_VERSION


class LatentModelService:
    """Validate and execute requests against one fixed latent text model."""

    def __init__(self, model: LatentTextModel, model_id: str) -> None:
        self.model = model
        self.model_id = model_id

    def handle(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_metadata(payload)
        if operation == "encode":
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            prompt = self._prompt(payload)
            result = {
                "values": self.model.text_to_embedding(
                    prepend_prompt(text, prompt),
                    canvas_length=self._canvas_length(payload),
                    multi_canvas=self._multi_canvas(payload),
                ).values.float().tolist()
            }
        elif operation == "decode":
            try:
                embedding = TextEmbedding(
                    torch.tensor(payload["values"], dtype=torch.float32)
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("values must be a finite embedding matrix") from error
            result = {
                "text": self.model.embedding_to_text(
                    embedding,
                    canvas_length=self._canvas_length(payload),
                    multi_canvas=self._multi_canvas(payload),
                )
            }
        elif operation == "interpolate":
            texts = payload.get("texts")
            if (
                not isinstance(texts, list)
                or not texts
                or not all(isinstance(text, str) for text in texts)
            ):
                raise ValueError("texts must be a non-empty list of strings")
            result = {
                "text": interpolate_texts(
                    self.model,
                    *texts,
                    prompt=self._prompt(payload),
                    canvas_length=self._canvas_length(payload),
                    multi_canvas=self._multi_canvas(payload),
                )
            }
        else:
            raise ValueError(f"unsupported operation: {operation}")

        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": self.model_id,
            **result,
        }

    def _validate_metadata(self, payload: dict[str, Any]) -> None:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol version does not match")
        if payload.get("model_id") != self.model_id:
            raise ValueError("model ID does not match")

    @staticmethod
    def _prompt(payload: dict[str, Any]) -> str | None:
        prompt = payload.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        return prompt

    @staticmethod
    def _canvas_length(payload: dict[str, Any]) -> int | None:
        canvas_length = payload.get("canvas_length")
        if canvas_length is not None and (
            not isinstance(canvas_length, int)
            or isinstance(canvas_length, bool)
            or canvas_length <= 0
        ):
            raise ValueError("canvas_length must be a positive integer")
        return canvas_length

    @staticmethod
    def _multi_canvas(payload: dict[str, Any]) -> int | None:
        multi_canvas = payload.get("multi_canvas")
        if multi_canvas is not None and (
            not isinstance(multi_canvas, int)
            or isinstance(multi_canvas, bool)
            or multi_canvas <= 0
        ):
            raise ValueError("multi_canvas must be a positive integer")
        return multi_canvas


def create_app(service: LatentModelService):
    """Create a FastAPI application without requiring FastAPI for local use."""

    fastapi = import_module("fastapi")
    FastAPI = fastapi.FastAPI
    HTTPException = fastapi.HTTPException

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": service.model_id,
        }

    @app.post("/{operation}")
    def execute(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.handle(operation, payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app