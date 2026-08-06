import pytest
import torch

from mean_idea.diffusion_gemma import _highest_score_token_ids


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
