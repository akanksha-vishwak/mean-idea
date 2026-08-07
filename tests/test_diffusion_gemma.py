import pytest
import torch

from mean_idea.diffusion_gemma import (
    _highest_score_token_ids,
    _validate_max_input_tokens,
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


def test_validate_max_input_tokens_accepts_multiple_canvases() -> None:
    assert (
        _validate_max_input_tokens(
            1024,
            canvas_length=256,
            context_length=262144,
            prompt_length=20,
        )
        == 1024
    )


@pytest.mark.parametrize("max_input_tokens", [0, 255, 257])
def test_validate_max_input_tokens_rejects_invalid_lengths(
    max_input_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        _validate_max_input_tokens(
            max_input_tokens,
            canvas_length=256,
            context_length=262144,
            prompt_length=20,
        )


def test_validate_max_input_tokens_rejects_context_overflow() -> None:
    with pytest.raises(ValueError, match="maximum is 261888"):
        _validate_max_input_tokens(
            262144,
            canvas_length=256,
            context_length=262144,
            prompt_length=20,
        )
