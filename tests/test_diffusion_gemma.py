import pytest
import torch

from mean_idea.diffusion_gemma import (
    _generate_canvas,
    _highest_score_token_ids,
    _input_length,
    _sample_tokens_and_entropy,
    _token_entropy,
    _validate_max_iterations,
)


def test_highest_score_token_ids_uses_tied_lm_head_across_chunks() -> None:
    vocabulary = torch.tensor(
        [
            [10.0, 0.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [-2.0, -2.0],
        ]
    )
    values = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

    result = _highest_score_token_ids(values, vocabulary, chunk_size=2)

    assert torch.equal(result, torch.tensor([0, 3]))


def test_highest_score_token_ids_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        _highest_score_token_ids(
            torch.zeros(1, 2), torch.zeros(1, 2), chunk_size=0
        )


def test_token_entropy_uses_tied_lm_head_across_chunks() -> None:
    vocabulary = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    values = torch.tensor([[2.0, 1.0], [0.0, 0.0]])

    result = _token_entropy(values, vocabulary, chunk_size=2)
    expected = torch.distributions.Categorical(
        logits=values @ vocabulary.T
    ).entropy()

    assert torch.allclose(result, expected)


def test_sample_tokens_and_entropy_chunks_sequence() -> None:
    torch.manual_seed(7)
    logits = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )

    sampled, entropy = _sample_tokens_and_entropy(
        logits,
        sequence_chunk_size=2,
    )
    expected_entropy = torch.distributions.Categorical(
        logits=logits
    ).entropy()

    assert sampled.shape == (1, 3)
    assert torch.allclose(entropy, expected_entropy)


def test_sample_tokens_and_entropy_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        _sample_tokens_and_entropy(
            torch.zeros(1, 1, 2),
            sequence_chunk_size=0,
        )


def test_generate_canvas_skips_model_for_zero_iterations() -> None:
    class Model:
        def generate(self, **kwargs):
            raise AssertionError("generate must not be called")

    decoder_input_ids = torch.tensor([[3, 2, 1]])

    result = _generate_canvas(
        Model(),
        inputs={"input_ids": torch.tensor([[9, 8]])},
        decoder_input_ids=decoder_input_ids,
        canvas_length=3,
        max_iterations=0,
    )

    assert result is decoder_input_ids


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_validate_max_iterations_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _validate_max_iterations(value)


def test_input_length_accepts_multiple_canvases() -> None:
    assert (
        _input_length(
            canvas_length=256,
            multi_canvas=4,
            context_length=262144,
            prompt_length=20,
        )
        == 1024
    )


@pytest.mark.parametrize("multi_canvas", [0, -1, True])
def test_input_length_rejects_invalid_multi_canvas(
    multi_canvas: int,
) -> None:
    with pytest.raises(ValueError):
        _input_length(
            canvas_length=256,
            multi_canvas=multi_canvas,
            context_length=262144,
            prompt_length=20,
        )


def test_input_length_rejects_context_overflow() -> None:
    with pytest.raises(ValueError, match="maximum is 1023"):
        _input_length(
            canvas_length=256,
            multi_canvas=1024,
            context_length=262144,
            prompt_length=20,
        )


def test_input_length_rejects_invalid_canvas_length() -> None:
    with pytest.raises(ValueError, match="canvas_length"):
        _input_length(
            canvas_length=0,
            multi_canvas=1,
            context_length=262144,
            prompt_length=20,
        )
