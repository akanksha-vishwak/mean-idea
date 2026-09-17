from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
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


EXPERIMENT_ID = "001-code-state-controls"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
SOURCE_A_PATH = REPOSITORY_ROOT / "tests" / "example" / "sort-agent-013.py"
SOURCE_B_PATH = REPOSITORY_ROOT / "tests" / "example" / "sort-agent-014.py"
PROMPT_PATH = REPOSITORY_ROOT / "tests" / "example" / "prompt.txt"
GENERATION_PROMPT = (
    "Decode the supplied canvas into one complete Python implementation of "
    "my_sort. Preserve the specific algorithmic and implementation details "
    "present in the canvas. Do not use list.sort or sorted. Return only Python "
    "code without Markdown fences or explanation."
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
            "Run Experiment 001: A+A and B+B source-recovery controls followed "
            "by the A+B entropy-weighted code-state mixture."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=256)
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


def token_count(model: DiffusionGemma, text: str) -> int:
    encoded = model.processor.tokenizer(text, add_special_tokens=True)
    return len(encoded.input_ids)


def _source_metadata(
    *,
    model: DiffusionGemma,
    name: str,
    path: Path,
    text: str,
    canvas_length: int,
) -> dict[str, object]:
    count = token_count(model, text)
    return {
        "id": name,
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
        "token_count": count,
        "truncated": count > canvas_length,
    }


def syntax_result(text: str) -> dict[str, object]:
    try:
        ast.parse(text)
    except SyntaxError as error:
        return {
            "parseable_python": False,
            "syntax_error": {
                "message": error.msg,
                "line": error.lineno,
                "offset": error.offset,
            },
        }
    return {"parseable_python": True, "syntax_error": None}


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


def print_generated_text(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    printable = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(printable, flush=True)


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

    sources = {
        "A": SOURCE_A_PATH.read_text(encoding="utf-8"),
        "B": SOURCE_B_PATH.read_text(encoding="utf-8"),
    }
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Canvas length: {args.canvas_length}", flush=True)
    print(f"Denoising steps: {args.steps}", flush=True)
    print(f"Source A: {SOURCE_A_PATH.relative_to(REPOSITORY_ROOT)}", flush=True)
    print(f"Source B: {SOURCE_B_PATH.relative_to(REPOSITORY_ROOT)}", flush=True)
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

    source_metadata = {
        name: _source_metadata(
            model=model,
            name=name,
            path=path,
            text=sources[name],
            canvas_length=args.canvas_length,
        )
        for name, path in (("A", SOURCE_A_PATH), ("B", SOURCE_B_PATH))
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
        "sources": source_metadata,
        "shared_source_prompt": {
            "path": str(PROMPT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": file_sha256(PROMPT_PATH),
        },
        "runs": runs,
    }
    write_results(results_path, results)

    for condition_id, left_name, right_name in CONDITIONS:
        print(
            f"\n=== {condition_id}: {left_name}+{right_name} ===",
            flush=True,
        )
        set_seed(args.seed)
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        try:
            text = interpolate_texts(
                model,
                sources[left_name],
                sources[right_name],
                prompts=(prompt, prompt),
                canvas_length=args.canvas_length,
                multi_canvas=1,
                max_iterations=args.steps,
            )
        except Exception as error:
            run = {
                "condition": condition_id,
                "sources": [left_name, right_name],
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
            "sources": [left_name, right_name],
            "status": "completed",
            "started_at": started_at,
            "duration_seconds": round(duration, 3),
            "output_file": output_path.name,
            "text": text,
            **syntax_result(text),
            "lexical_indicators": lexical_indicators(text),
        }
        runs.append(run)
        write_results(results_path, results)
        print_generated_text(text)

    results["status"] = "completed"
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"\nSaved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
