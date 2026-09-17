from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from mean_idea.api import interpolate_texts
from mean_idea.diffusion_gemma import (
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "002-prompt-state-controls"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PROMPT_A_PATH = EXPERIMENT_DIR / "prompt-a.txt"
PROMPT_B_PATH = EXPERIMENT_DIR / "prompt-b.txt"
GENERATION_PROMPT = (
    "Generate one concise optimization idea for the Python function my_sort. "
    "Preserve the specific algorithmic and implementation details represented "
    "in the supplied canvas. Do not use list.sort or sorted. Return only the "
    "optimization idea."
)
CONDITIONS = (
    ("a-control", "A", "A"),
    ("b-control", "B", "B"),
    ("a-b-mixture", "A", "B"),
)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 002: blank-canvas A+A and B+B prompt controls "
            "followed by the A+B entropy-weighted prompt-state mixture."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=64)
    parser.add_argument("--steps", type=non_negative_integer, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_token_ids(model: DiffusionGemma, prompt: str) -> list[int]:
    content = f"{prompt}\n\n{GENERATION_PROMPT}"
    encoded = model.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return encoded["input_ids"][0].tolist()


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        count += 1
    return count


def lexical_indicators(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "mentions_counting_sort": "counting sort" in lowered,
        "mentions_frequency": "frequen" in lowered or "count[" in lowered,
        "mentions_known_range": "10000" in lowered or "9999" in lowered,
        "mentions_numpy": "numpy" in lowered or "np." in lowered,
        "mentions_bincount": "bincount" in lowered,
        "mentions_repeat": "repeat" in lowered,
        "mentions_quicksort": "quicksort" in lowered or "quick_sort" in lowered,
        "uses_prohibited_builtin": "list.sort" in lowered or "sorted(" in lowered,
    }


def write_results(path: Path, results: dict[str, object]) -> None:
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    results_path = output_dir / "results.json"
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    canvas = BLANK_CANVAS_PATH.read_text(encoding="utf-8")
    if canvas:
        raise ValueError(f"{BLANK_CANVAS_PATH} must be empty")
    prompts = {
        "A": PROMPT_A_PATH.read_text(encoding="utf-8"),
        "B": PROMPT_B_PATH.read_text(encoding="utf-8"),
    }

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Canvas: empty, {args.canvas_length} positions", flush=True)
    print(f"Denoising steps: {args.steps}", flush=True)
    print(f"Prompt A: {PROMPT_A_PATH.relative_to(REPOSITORY_ROOT)}", flush=True)
    print(f"Prompt B: {PROMPT_B_PATH.relative_to(REPOSITORY_ROOT)}", flush=True)
    print("Loading the model once for all three conditions...", flush=True)

    model = DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model_id,
            prompt=GENERATION_PROMPT,
            max_denoising_steps=args.steps,
            canvas_length=args.canvas_length,
            multi_canvas=1,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    )
    prompt_ids = {
        name: prompt_token_ids(model, prompt)
        for name, prompt in prompts.items()
    }
    runs: list[dict[str, object]] = []
    results: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "settings": {
            "canvas_length": args.canvas_length,
            "multi_canvas": 1,
            "max_denoising_steps": args.steps,
            "seed_per_condition": args.seed,
            "mixing": "original entropy-weighted per-position mean",
            "generation_prompt": GENERATION_PROMPT,
        },
        "canvas": {
            "path": str(BLANK_CANVAS_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": file_sha256(BLANK_CANVAS_PATH),
            "character_count": len(canvas),
        },
        "prompts": {
            name: {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": file_sha256(path),
                "token_count_with_generation_prompt": len(prompt_ids[name]),
            }
            for name, path in (("A", PROMPT_A_PATH), ("B", PROMPT_B_PATH))
        },
        "prompt_common_prefix_tokens": common_prefix_length(
            prompt_ids["A"], prompt_ids["B"]
        ),
        "runs": runs,
    }
    write_results(results_path, results)

    for condition_id, left_name, right_name in CONDITIONS:
        print(
            f"\n=== {condition_id}: Prompt {left_name}+{right_name} ===",
            flush=True,
        )
        set_seed(args.seed)
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        try:
            text = interpolate_texts(
                model,
                canvas,
                canvas,
                prompts=(prompts[left_name], prompts[right_name]),
                canvas_length=args.canvas_length,
                multi_canvas=1,
                max_iterations=args.steps,
            )
        except Exception as error:
            run = {
                "condition": condition_id,
                "prompts": [left_name, right_name],
                "status": "failed",
                "started_at": started_at,
                "error": f"{type(error).__name__}: {error}",
            }
            runs.append(run)
            results["status"] = "failed"
            write_results(results_path, results)
            raise

        duration = time.monotonic() - started
        output_path = output_dir / f"{condition_id}.txt"
        output_path.write_text(text + "\n", encoding="utf-8")
        run = {
            "condition": condition_id,
            "prompts": [left_name, right_name],
            "status": "completed",
            "started_at": started_at,
            "duration_seconds": round(duration, 3),
            "output_file": output_path.name,
            "lexical_indicators": lexical_indicators(text),
        }
        runs.append(run)
        write_results(results_path, results)
        print(
            f"Completed in {run['duration_seconds']}s; saved raw output to "
            f"{output_path}",
            flush=True,
        )

    results["status"] = "completed"
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"\nSaved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
