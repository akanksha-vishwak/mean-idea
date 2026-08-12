from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen

import torch

from mean_idea.api import TextEmbedding

PROTOCOL_VERSION = 6


@dataclass(frozen=True, slots=True)
class RemoteModelSettings:
    endpoint_url: str
    model_id: str
    api_token: str | None = None
    timeout: float = 120.0


class RemoteLatentTextModel:
    """HTTP client for a remotely hosted latent text model."""

    def __init__(
        self,
        settings: RemoteModelSettings,
        *,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport or self._post

    def text_to_embedding(
        self,
        text: str,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding:
        payload = {"text": text}
        if prompt is not None:
            payload["prompt"] = prompt
        if canvas_length is not None:
            payload["canvas_length"] = canvas_length
        if multi_canvas is not None:
            payload["multi_canvas"] = multi_canvas
        response = self._request("encode", payload)
        try:
            values = torch.tensor(response["values"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("remote encode response has invalid embedding values") from error
        return TextEmbedding(values)

    def embedding_to_text(
        self,
        embedding: TextEmbedding,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
        max_iterations: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "values": embedding.values.detach().float().cpu().tolist()
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if canvas_length is not None:
            payload["canvas_length"] = canvas_length
        if multi_canvas is not None:
            payload["multi_canvas"] = multi_canvas
        if max_iterations is not None:
            payload["max_iterations"] = max_iterations
        response = self._request(
            "decode",
            payload,
        )
        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError("remote decode response has invalid text")
        return text

    def interpolate_texts(
        self,
        *texts: str,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
        max_iterations: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {"texts": list(texts)}
        if prompt is not None:
            payload["prompt"] = prompt
        if canvas_length is not None:
            payload["canvas_length"] = canvas_length
        if multi_canvas is not None:
            payload["multi_canvas"] = multi_canvas
        if max_iterations is not None:
            payload["max_iterations"] = max_iterations
        response = self._request("interpolate", payload)
        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError("remote interpolate response has invalid text")
        return text

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._transport(
            operation,
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_id": self.settings.model_id,
                **payload,
            },
        )
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("remote protocol version does not match")
        if response.get("model_id") != self.settings.model_id:
            raise ValueError("remote model ID does not match")
        return response

    def _post(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.endpoint_url.rstrip('/')}/{operation}"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_token:
            headers["Authorization"] = f"Bearer {self.settings.api_token}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("remote response must be a JSON object")
        return result