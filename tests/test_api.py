import pytest
import torch

from mean_idea.api import (
    TextEmbedding,
    interpolate_texts,
    mean_embeddings,
)


class FakeModel:
    def text_to_embedding(
        self,
        text: str,
        *,
        prompt: str | None = None,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding:
        return TextEmbedding(torch.tensor([[float(len(text)), 1.0]]))

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


def test_mean_embeddings() -> None:
    left = TextEmbedding(torch.tensor([[1.0, 3.0]]))
    right = TextEmbedding(torch.tensor([[3.0, 5.0]]))

    result = mean_embeddings(left, right)

    assert torch.equal(result.values, torch.tensor([[2.0, 4.0]]))
    assert torch.equal(result.noise, torch.tensor([0.0]))


def test_mean_embeddings_weights_lower_noise_more_heavily() -> None:
    left = TextEmbedding(torch.tensor([[1.0, 3.0]]), torch.tensor([0.0]))
    right = TextEmbedding(torch.tensor([[3.0, 5.0]]), torch.tensor([2.0]))

    result = mean_embeddings(left, right)
    expected_weights = torch.softmax(torch.tensor([0.0, -2.0]), dim=0)

    assert result.noise is not None
    assert torch.allclose(
        result.values,
        expected_weights[0] * left.values + expected_weights[1] * right.values,
    )
    assert torch.allclose(result.noise, expected_weights[1:] * 2.0)


def test_text_embedding_rejects_invalid_noise_shape() -> None:
    with pytest.raises(ValueError, match="noise must have shape"):
        TextEmbedding(torch.zeros(2, 3), torch.zeros(2, 1))


def test_mean_embeddings_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_embeddings(
            TextEmbedding(torch.zeros(2, 3)),
            TextEmbedding(torch.zeros(3, 3)),
        )


def test_interpolate_texts_uses_two_call_api() -> None:
    assert interpolate_texts(FakeModel(), "a", "abc") == "[[2.0, 1.0]]"


def test_interpolate_texts_passes_individual_prompts_only_to_encoders() -> None:
    seen_prompts = iter(("left prompt", "right prompt"))

    class PromptModel(FakeModel):
        def text_to_embedding(
            self,
            text: str,
            *,
            prompt: str | None = None,
            canvas_length: int | None = None,
            multi_canvas: int | None = None,
        ) -> TextEmbedding:
            assert prompt == next(seen_prompts)
            return super().text_to_embedding(text)

        def embedding_to_text(
            self,
            embedding: TextEmbedding,
            *,
            prompt: str | None = None,
            canvas_length: int | None = None,
            multi_canvas: int | None = None,
            max_iterations: int | None = None,
        ) -> str:
            assert prompt is None
            assert max_iterations is None
            return super().embedding_to_text(embedding)

    assert (
        interpolate_texts(
            PromptModel(),
            "a",
            "abc",
            prompts=("left prompt", "right prompt"),
        )
        == "[[2.0, 1.0]]"
    )


def test_interpolate_texts_rejects_mismatched_prompts() -> None:
    with pytest.raises(ValueError, match="same length"):
        interpolate_texts(FakeModel(), "a", "b", prompts=("only one",))
    with pytest.raises(ValueError, match="sequence"):
        interpolate_texts(FakeModel(), "a", "b", prompts="ab")


def test_interpolate_texts_passes_canvas_length() -> None:
    class CanvasModel(FakeModel):
        def text_to_embedding(
            self,
            text: str,
            *,
            prompt: str | None = None,
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
            prompt: str | None = None,
            canvas_length: int | None = None,
            multi_canvas: int | None = None,
            max_iterations: int | None = None,
        ) -> str:
            assert canvas_length == 512
            assert multi_canvas == 2
            assert max_iterations == 0
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
        max_iterations=0,
    )
