from pathlib import Path

import pytest

from mean_idea.cli import build_model, build_parser
from mean_idea.diffusion_gemma import DEFAULT_MODEL_ID, DEFAULT_PROMPT
from mean_idea.remote import RemoteLatentTextModel


def test_cli_defaults_are_strings() -> None:
    args = build_parser().parse_args(["left.py", "right.py"])

    assert args.model_id == DEFAULT_MODEL_ID
    assert args.prompt is None
    assert args.generation_prompt == DEFAULT_PROMPT
    assert args.canvas_length == 256
    assert args.multi_canvas == 1
    assert args.steps == 48


def test_cli_accepts_prompt_file() -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", "--prompt", "prompt.txt"]
    )

    assert args.prompt == Path("prompt.txt")


@pytest.mark.parametrize("option", ["--model-id", "--model"])
def test_cli_accepts_model_id(option: str) -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", option, "example/model"]
    )

    assert args.model_id == "example/model"


def test_cli_accepts_multi_canvas() -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", "--multi-canvas", "4"]
    )

    assert args.multi_canvas == 4


def test_cli_accepts_canvas_length() -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", "--canvas-length", "1024"]
    )

    assert args.canvas_length == 1024


def test_cli_accepts_zero_steps() -> None:
    args = build_parser().parse_args(
        ["left.py", "right.py", "--steps", "0"]
    )

    assert args.steps == 0


def test_cli_uses_steps_for_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = None

    class Model:
        def __init__(self, settings):
            nonlocal captured
            captured = settings

    monkeypatch.setattr("mean_idea.cli.DiffusionGemma", Model)
    args = build_parser().parse_args(
        ["left.py", "right.py", "--steps", "0"]
    )

    build_model(args)

    assert captured.max_denoising_steps == 0


def test_cli_rejects_negative_steps() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["left.py", "right.py", "--steps", "-1"]
        )


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
