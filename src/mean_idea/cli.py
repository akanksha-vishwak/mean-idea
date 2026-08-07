from __future__ import annotations

import argparse
from pathlib import Path

from mean_idea.api import interpolate_texts
from mean_idea.diffusion_gemma import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode the mean DiffusionGemma embedding of two texts."
    )
    parser.add_argument("first", type=Path, help="first UTF-8 text file")
    parser.add_argument("second", type=Path, help="second UTF-8 text file")
    parser.add_argument("-o", "--output", type=Path, help="write output to this file")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model,
            prompt=args.prompt,
            max_denoising_steps=args.steps,
        )
    )
    result = interpolate_texts(
        model,
        args.first.read_text(encoding="utf-8"),
        args.second.read_text(encoding="utf-8"),
    )

    if args.output:
        args.output.write_text(f"{result}\n", encoding="utf-8")
    else:
        print(result)
