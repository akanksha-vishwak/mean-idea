import pytest
import torch

from mean_idea.api import TextEmbedding
from mean_idea.remote import PROTOCOL_VERSION
from mean_idea.service import LatentModelService


class FakeModel:
    def text_to_embedding(
        self,
        text: str,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding:
        return TextEmbedding(
            torch.tensor(
                [
                    [
                        float(len(text)),
                        float(len(prompt or "")),
                        float(canvas_length or 1),
                    ]
                ]
            ),
            torch.tensor([0.0]),
        )

    def embedding_to_text(
        self,
        embedding: TextEmbedding,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
        max_iterations: int | None = None,
    ) -> str:
        return str(embedding.values.tolist())


def request(**payload):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "example/model",
        **payload,
    }


def test_service_executes_latent_operations() -> None:
    service = LatentModelService(FakeModel(), "example/model")

    encoded = service.handle(
        "encode",
        request(
            text="abc",
            prompt="prompt: ",
            canvas_length=512,
            multi_canvas=2,
        ),
    )
    decoded = service.handle(
        "decode",
        request(
            values=[[2.0, 1.0]],
            prompt="prompt: ",
            max_iterations=0,
        ),
    )
    interpolated = service.handle(
        "interpolate",
        request(
            texts=["a", "abc"],
            prompt="prompt: ",
            canvas_length=512,
            multi_canvas=2,
            max_iterations=24,
        ),
    )

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "example/model",
    }
    assert encoded == {
        **metadata,
        "values": [[3.0, 8.0, 512.0]],
        "noise": [0.0],
    }
    assert decoded == {**metadata, "text": "[[2.0, 1.0]]"}
    assert interpolated == {
        **metadata,
        "text": "[[2.0, 8.0, 512.0]]",
    }


@pytest.mark.parametrize(
    ("operation", "payload", "message"),
    [
        ("encode", request(text=1), "text must"),
        ("encode", request(text="abc", prompt=1), "prompt must"),
        ("decode", request(values=[1.0]), "embedding matrix"),
        ("interpolate", request(texts=[]), "non-empty"),
        ("interpolate", request(texts=["abc"], prompt=[]), "prompt must"),
        ("interpolate", request(texts=["abc"], canvas_length=0), "canvas_length"),
        ("interpolate", request(texts=["abc"], canvas_length=True), "canvas_length"),
        ("interpolate", request(texts=["abc"], multi_canvas=0), "multi_canvas"),
        ("interpolate", request(texts=["abc"], multi_canvas=True), "multi_canvas"),
        ("interpolate", request(texts=["abc"], max_iterations=-1), "max_iterations"),
        ("interpolate", request(texts=["abc"], max_iterations=True), "max_iterations"),
        ("unknown", request(), "unsupported operation"),
    ],
)
def test_service_rejects_invalid_requests(operation, payload, message) -> None:
    service = LatentModelService(FakeModel(), "example/model")

    with pytest.raises(ValueError, match=message):
        service.handle(operation, payload)