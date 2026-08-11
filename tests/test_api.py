import pytest
import torch

from mean_idea.api import (
    TextEmbedding,
    interpolate_texts,
    mean_embeddings,
    prepend_prompt,
)


class FakeModel:
    def text_to_embedding(
        self,
        text: str,
        *,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding:
        return TextEmbedding(torch.tensor([[float(len(text)), 1.0]]))

    def embedding_to_text(
        self,
        embedding: TextEmbedding,
        *,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> str:
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


def test_interpolate_texts_prepends_prompt_to_each_text() -> None:
    assert (
        interpolate_texts(FakeModel(), "a", "abc", prompt="prompt: ")
        == "[[10.0, 1.0]]"
    )


def test_prepend_prompt_preserves_text_without_prompt() -> None:
    assert prepend_prompt("source") == "source"


def test_interpolate_texts_passes_canvas_length() -> None:
    class CanvasModel(FakeModel):
        def text_to_embedding(
            self,
            text: str,
            *,
            canvas_length: int | None = None,
            multi_canvas: int | None = None,
        ) -> TextEmbedding:
            assert canvas_length == 512
            assert multi_canvas == 2
            return super().text_to_embedding(
                text,
                canvas_length=canvas_length,
                multi_canvas=multi_canvas,
            )

        def embedding_to_text(
            self,
            embedding: TextEmbedding,
            *,
            canvas_length: int | None = None,
            multi_canvas: int | None = None,
        ) -> str:
            assert canvas_length == 512
            assert multi_canvas == 2
            return super().embedding_to_text(
                embedding,
                canvas_length=canvas_length,
                multi_canvas=multi_canvas,
            )

    interpolate_texts(
        CanvasModel(),
        "a",
        "b",
        canvas_length=512,
        multi_canvas=2,
    )
