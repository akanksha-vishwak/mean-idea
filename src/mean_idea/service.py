from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from mean_idea.api import LatentTextModel, TextEmbedding, interpolate_texts
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
            result = {
                "values": self.model.text_to_embedding(text).values.float().tolist()
            }
        elif operation == "decode":
            try:
                embedding = TextEmbedding(
                    torch.tensor(payload["values"], dtype=torch.float32)
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("values must be a finite embedding matrix") from error
            result = {"text": self.model.embedding_to_text(embedding)}
        elif operation == "interpolate":
            texts = payload.get("texts")
            if (
                not isinstance(texts, list)
                or not texts
                or not all(isinstance(text, str) for text in texts)
            ):
                raise ValueError("texts must be a non-empty list of strings")
            result = {"text": interpolate_texts(self.model, *texts)}
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