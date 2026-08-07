import pytest
import torch

from mean_idea.api import TextEmbedding
from mean_idea.remote import PROTOCOL_VERSION
from mean_idea.service import LatentModelService


class FakeModel:
    def text_to_embedding(self, text: str) -> TextEmbedding:
        return TextEmbedding(torch.tensor([[float(len(text)), 1.0]]))

    def embedding_to_text(self, embedding: TextEmbedding) -> str:
        return str(embedding.values.tolist())


def request(**payload):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "example/model",
        **payload,
    }


def test_service_executes_latent_operations() -> None:
    service = LatentModelService(FakeModel(), "example/model")

    encoded = service.handle("encode", request(text="abc"))
    decoded = service.handle("decode", request(values=[[2.0, 1.0]]))
    interpolated = service.handle(
        "interpolate", request(texts=["a", "abc"])
    )

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "example/model",
    }
    assert encoded == {**metadata, "values": [[3.0, 1.0]]}
    assert decoded == {**metadata, "text": "[[2.0, 1.0]]"}
    assert interpolated == {**metadata, "text": "[[2.0, 1.0]]"}


@pytest.mark.parametrize(
    ("operation", "payload", "message"),
    [
        ("encode", request(text=1), "text must"),
        ("decode", request(values=[1.0]), "embedding matrix"),
        ("interpolate", request(texts=[]), "non-empty"),
        ("unknown", request(), "unsupported operation"),
    ],
)
def test_service_rejects_invalid_requests(operation, payload, message) -> None:
    service = LatentModelService(FakeModel(), "example/model")

    with pytest.raises(ValueError, match=message):
        service.handle(operation, payload)