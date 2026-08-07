import pytest

from mean_idea.cli import build_model, build_parser
from mean_idea.diffusion_gemma import DEFAULT_MODEL_ID, DEFAULT_PROMPT
from mean_idea.remote import RemoteLatentTextModel


def test_cli_defaults_are_strings() -> None:
    args = build_parser().parse_args(["left.py", "right.py"])

    assert args.model_id == DEFAULT_MODEL_ID
    assert args.prompt == DEFAULT_PROMPT


@pytest.mark.parametrize("option", ["--model-id", "--model"])
def test_cli_accepts_model_id(option: str) -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", option, "example/model"]
    )

    assert args.model_id == "example/model"


def test_cli_builds_remote_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_TOKEN", "secret")
    args = build_parser().parse_args(
        [
            "left.py",
            "right.py",
            "--backend",
            "remote",
            "--endpoint-url",
            "https://example.test/api",
            "--api-token-env",
            "TEST_API_TOKEN",
            "--model-id",
            "example/model",
        ]
    )

    model = build_model(args)

    assert isinstance(model, RemoteLatentTextModel)
    assert model.settings.endpoint_url == "https://example.test/api"
    assert model.settings.api_token == "secret"


def test_cli_requires_remote_endpoint() -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", "--backend", "remote"]
    )

    with pytest.raises(ValueError, match="--endpoint-url"):
        build_model(args)
