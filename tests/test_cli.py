from mean_idea.cli import build_parser
from mean_idea.diffusion_gemma import DEFAULT_MODEL_ID, DEFAULT_PROMPT


def test_cli_defaults_are_strings() -> None:
    args = build_parser().parse_args(["left.py", "right.py"])

    assert args.model == DEFAULT_MODEL_ID
    assert args.prompt == DEFAULT_PROMPT
