from __future__ import annotations

import argparse
import os
from pathlib import Path

from mean_idea.api import LatentTextModel, interpolate_texts
from mean_idea.diffusion_gemma import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    DiffusionGemma,
    DiffusionGemmaSettings,
)
from mean_idea.remote import RemoteLatentTextModel, RemoteModelSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode the mean DiffusionGemma embedding of two texts."
    )
    parser.add_argument("first", type=Path, help="first UTF-8 text file")
    parser.add_argument("second", type=Path, help="second UTF-8 text file")
    parser.add_argument("-o", "--output", type=Path, help="write output to this file")
    parser.add_argument(
        "--backend",
        choices=("local", "remote"),
        default="local",
        help="run the model locally or call a remote latent-model endpoint",
    )
    parser.add_argument(
        "--endpoint-url",
        help="remote endpoint base URL (required with --backend remote)",
    )
    parser.add_argument(
        "--api-token-env",
        default="MEAN_IDEA_API_TOKEN",
        help="environment variable containing the remote API token",
    )
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face DiffusionGemma model ID",
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=256,
        help="fixed input length, in multiples of the model's 256-token canvas",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def build_model(args: argparse.Namespace) -> LatentTextModel:
    if args.backend == "remote":
        if not args.endpoint_url:
            raise ValueError("--endpoint-url is required with --backend remote")
        return RemoteLatentTextModel(
            RemoteModelSettings(
                endpoint_url=args.endpoint_url,
                model_id=args.model_id,
                api_token=os.environ.get(args.api_token_env),
            )
        )

    return DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model_id,
            prompt=args.prompt,
            max_denoising_steps=args.steps,
            max_input_tokens=args.max_input_tokens,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        model = build_model(args)
    except ValueError as error:
        parser.error(str(error))
    texts = (
        args.first.read_text(encoding="utf-8"),
        args.second.read_text(encoding="utf-8"),
    )
    if isinstance(model, RemoteLatentTextModel):
        result = model.interpolate_texts(*texts)
    else:
        result = interpolate_texts(model, *texts)

    if args.output:
        args.output.write_text(f"{result}\n", encoding="utf-8")
    else:
        print(result)
