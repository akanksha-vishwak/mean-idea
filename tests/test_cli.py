from pathlib import Path

import pytest

from mean_idea.cli import build_model, build_parser, main
from mean_idea.diffusion_gemma import DEFAULT_MODEL_ID, DEFAULT_PROMPT
from mean_idea.remote import RemoteLatentTextModel

INPUT_ARGS = [
    "--canvas1",
    "left.py",
    "--prompt1",
    "left-prompt.txt",
    "--canvas2",
    "right.py",
    "--prompt2",
    "right-prompt.txt",
]


def test_cli_defaults_are_strings() -> None:
    args = build_parser().parse_args(INPUT_ARGS)

    assert args.model_id == DEFAULT_MODEL_ID
    assert args.canvas1 == Path("left.py")
    assert args.prompt1 == Path("left-prompt.txt")
    assert args.canvas2 == Path("right.py")
    assert args.prompt2 == Path("right-prompt.txt")
    assert args.generation_prompt == DEFAULT_PROMPT
    assert args.canvas_length == 256
    assert args.multi_canvas == 1
    assert args.steps == 48


@pytest.mark.parametrize("option", ["--model-id", "--model"])
def test_cli_accepts_model_id(option: str) -> None:
    args = build_parser().parse_args(
        [*INPUT_ARGS, option, "example/model"]
    )

    assert args.model_id == "example/model"


def test_cli_accepts_multi_canvas() -> None:
    args = build_parser().parse_args(
        [*INPUT_ARGS, "--multi-canvas", "4"]
    )

    assert args.multi_canvas == 4


def test_cli_accepts_canvas_length() -> None:
    args = build_parser().parse_args(
        [*INPUT_ARGS, "--canvas-length", "1024"]
    )

    assert args.canvas_length == 1024


def test_cli_accepts_zero_steps() -> None:
    args = build_parser().parse_args(
        [*INPUT_ARGS, "--steps", "0"]
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
        [*INPUT_ARGS, "--steps", "0"]
    )

    build_model(args)

    assert captured.max_denoising_steps == 0


def test_cli_rejects_negative_steps() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [*INPUT_ARGS, "--steps", "-1"]
        )


def test_cli_builds_remote_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_TOKEN", "secret")
    args = build_parser().parse_args(
        [
            *INPUT_ARGS,
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
        [*INPUT_ARGS, "--backend", "remote"]
    )

    with pytest.raises(ValueError, match="--endpoint-url"):
        build_model(args)


def test_cli_requires_all_canvas_and_prompt_files() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--canvas1", "left.py"])


def test_cli_reads_each_canvas_and_prompt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = {
        "canvas1": tmp_path / "canvas1.txt",
        "prompt1": tmp_path / "prompt1.txt",
        "canvas2": tmp_path / "canvas2.txt",
        "prompt2": tmp_path / "prompt2.txt",
        "output": tmp_path / "output.txt",
    }
    paths["canvas1"].write_text("left canvas", encoding="utf-8")
    paths["prompt1"].write_text("left prompt", encoding="utf-8")
    paths["canvas2"].write_text("right canvas", encoding="utf-8")
    paths["prompt2"].write_text("right prompt", encoding="utf-8")
    captured = None

    def interpolate(model, *texts, **options):
        nonlocal captured
        captured = (texts, options)
        return "result"

    monkeypatch.setattr("mean_idea.cli.build_model", lambda args: object())
    monkeypatch.setattr("mean_idea.cli.interpolate_texts", interpolate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mean-idea",
            "--canvas1",
            str(paths["canvas1"]),
            "--prompt1",
            str(paths["prompt1"]),
            "--canvas2",
            str(paths["canvas2"]),
            "--prompt2",
            str(paths["prompt2"]),
            "--output",
            str(paths["output"]),
        ],
    )

    main()

    assert captured[0] == ("left canvas", "right canvas")
    assert captured[1]["prompts"] == ("left prompt", "right prompt")
    assert paths["output"].read_text(encoding="utf-8") == "result\n"
