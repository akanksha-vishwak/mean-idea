import pytest
import torch

from mean_idea.api import TextEmbedding, interpolate_texts, mean_embeddings


class FakeModel:
    def text_to_embedding(self, text: str) -> TextEmbedding:
        return TextEmbedding(torch.tensor([[float(len(text)), 1.0]]))

    def embedding_to_text(self, embedding: TextEmbedding) -> str:
        return str(embedding.values.tolist())


def test_mean_embeddings() -> None:
    left = TextEmbedding(torch.tensor([[1.0, 3.0]]))
    right = TextEmbedding(torch.tensor([[3.0, 5.0]]))

    result = mean_embeddings(left, right)

    assert torch.equal(result.values, torch.tensor([[2.0, 4.0]]))


def test_mean_embeddings_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_embeddings(
            TextEmbedding(torch.zeros(2, 3)),
            TextEmbedding(torch.zeros(3, 3)),
        )


def test_interpolate_texts_uses_two_call_api() -> None:
    assert interpolate_texts(FakeModel(), "a", "abc") == "[[2.0, 1.0]]"
