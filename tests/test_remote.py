from typing import Any

import pytest
import torch

from mean_idea.remote import (
    PROTOCOL_VERSION,
    RemoteLatentTextModel,
    RemoteModelSettings,
)


def test_remote_model_encodes_and_decodes() -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    def transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        requests.append((operation, payload))
        response: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": "example/model",
        }
        if operation == "encode":
            return {**response, "values": [[1.0, 2.0]]}
        if operation == "interpolate":
            return {**response, "text": "interpolated"}
        return {**response, "text": "decoded"}

    model = RemoteLatentTextModel(
        RemoteModelSettings("https://example.test", "example/model"),
        transport=transport,
    )

    embedding = model.text_to_embedding(
        "source", prompt="prompt: ", canvas_length=512, multi_canvas=2
    )
    result = model.embedding_to_text(
        embedding, canvas_length=512, multi_canvas=2
    )
    interpolated = model.interpolate_texts(
        "left",
        "right",
        prompt="prompt: ",
        canvas_length=512,
        multi_canvas=2,
    )

    assert torch.equal(embedding.values, torch.tensor([[1.0, 2.0]]))
    assert result == "decoded"
    assert interpolated == "interpolated"
    assert requests == [
        (
            "encode",
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_id": "example/model",
                "text": "source",
                "prompt": "prompt: ",
                "canvas_length": 512,
                "multi_canvas": 2,
            },
        ),
        (
            "decode",
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_id": "example/model",
                "values": [[1.0, 2.0]],
                "canvas_length": 512,
                "multi_canvas": 2,
            },
        ),
        (
            "interpolate",
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_id": "example/model",
                "texts": ["left", "right"],
                "prompt": "prompt: ",
                "canvas_length": 512,
                "multi_canvas": 2,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", PROTOCOL_VERSION + 1, "protocol version"),
        ("model_id", "other/model", "model ID"),
    ],
)
def test_remote_model_rejects_incompatible_server(
    field: str, value: Any, message: str
) -> None:
    response: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "example/model",
        "values": [[1.0]],
        field: value,
    }
    model = RemoteLatentTextModel(
        RemoteModelSettings("https://example.test", "example/model"),
        transport=lambda operation, payload: response,
    )

    with pytest.raises(ValueError, match=message):
        model.text_to_embedding("source")


def test_remote_model_rejects_malformed_embedding() -> None:
    model = RemoteLatentTextModel(
        RemoteModelSettings("https://example.test", "example/model"),
        transport=lambda operation, payload: {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": "example/model",
            "values": [1.0, 2.0],
        },
    )

    with pytest.raises(ValueError, match="shape"):
        model.text_to_embedding("source")