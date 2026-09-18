from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from mean_idea.api import TextEmbedding
from mean_idea.diffusion_gemma import (
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "004-controlled-prompt-weight-sweep"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PROMPT_A_PATH = EXPERIMENT_DIR / "prompt-a-counting.txt"
PROMPT_B_PATH = EXPERIMENT_DIR / "prompt-b-numba.txt"
GENERATION_PROMPT = (
    "Generate one concise optimization idea for the Python function my_sort. "
    "Preserve the specific algorithmic and implementation details represented "
    "in the supplied canvas. Do not use list.sort or sorted. Return only the "
    "optimization idea."
)
WEIGHTS = tuple(round(index / 10, 1) for index in range(11))
DEFAULT_SEEDS = (42, 43, 44)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 004: validate arithmetic interpolation endpoints, "
            "then sweep weights between counting-sort and Numba prompt states."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=64)
    parser.add_argument("--steps", type=positive_integer, default=8)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated generation seeds",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--allow-failed-controls",
        action="store_true",
        help="continue the sweep despite endpoint gate failure; for fixture testing only",
    )
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


def interpolate_embeddings(
    left: TextEmbedding,
    right: TextEmbedding,
    weight: float,
) -> TextEmbedding:
    if left.values.shape != right.values.shape:
        raise ValueError("source embeddings must have identical shapes")
    values = (1.0 - weight) * left.values.float() + weight * right.values.float()
    noise = None
    if left.noise is not None and right.noise is not None:
        noise = (1.0 - weight) * left.noise.float() + weight * right.noise.float()
    return TextEmbedding(values, noise)


def lexical_indicators(text: str) -> dict[str, bool]:
    lowered = text.lower()
    counting = "counting sort" in lowered
    frequency = "frequen" in lowered or "count[" in lowered
    bounded = (
        "bounded" in lowered
        or "10000" in lowered
        or "9999" in lowered
        or "known range" in lowered
        or "fixed-size" in lowered
        or "fixed size" in lowered
    )
    numba = "numba" in lowered or "@njit" in lowered
    jit = "@njit" in lowered or "jit" in lowered or "machine code" in lowered
    prohibited = "list.sort" in lowered or "sorted(" in lowered
    return {
        "mentions_counting_sort": counting,
        "mentions_frequency": frequency,
        "mentions_bounded_range": bounded,
        "retains_A": counting or (frequency and bounded),
        "mentions_numba": numba,
        "mentions_jit_compilation": jit,
        "retains_B": numba and jit,
        "retains_both": (counting or (frequency and bounded)) and numba and jit,
        "uses_prohibited_builtin": prohibited,
        "candidate_hybrid": (
            (counting or (frequency and bounded))
            and numba
            and jit
            and not prohibited
        ),
    }


def write_results(path: Path, results: dict[str, object]) -> None:
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_generation(
    *,
    model: DiffusionGemma,
    embedding: TextEmbedding,
    output_dir: Path,
    weight: float,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, object]:
    set_seed(seed)
    weight_label = f"{weight:.1f}".replace(".", "p")
    output_path = output_dir / f"weight-{weight_label}-seed-{seed}.txt"
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    text = model.embedding_to_text(
        embedding,
        canvas_length=canvas_length,
        multi_canvas=1,
        max_iterations=steps,
    )
    duration = time.monotonic() - started
    output_path.write_text(text + "\n", encoding="utf-8")
    return {
        "weight": weight,
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "output_file": output_path.name,
        "lexical_indicators": lexical_indicators(text),
    }


def controls_pass(runs: list[dict[str, object]]) -> bool:
    endpoint_a = [run for run in runs if run["weight"] == 0.0]
    endpoint_b = [run for run in runs if run["weight"] == 1.0]
    return bool(endpoint_a and endpoint_b) and all(
        bool(run["lexical_indicators"]["retains_A"])  # type: ignore[index]
        for run in endpoint_a
    ) and all(
        bool(run["lexical_indicators"]["retains_B"])  # type: ignore[index]
        for run in endpoint_b
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
    print(f"Weights: {WEIGHTS}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print("Loading the model once for all conditions...", flush=True)

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
    embeddings = {
        name: model.text_to_embedding(
            canvas,
            prompt=prompt,
            canvas_length=args.canvas_length,
            multi_canvas=1,
        )
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
            "weights": list(WEIGHTS),
            "seeds": list(args.seeds),
            "mixing": "global arithmetic final-canvas-state interpolation",
            "weight_meaning": "0.0 is all A; 1.0 is all B",
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
        "embedding_shapes": {
            name: list(embedding.values.shape)
            for name, embedding in embeddings.items()
        },
        "runs": runs,
    }
    write_results(results_path, results)

    endpoint_schedule = [
        (weight, seed)
        for weight in (0.0, 1.0)
        for seed in args.seeds
    ]
    for index, (weight, seed) in enumerate(endpoint_schedule, start=1):
        print(
            f"[control {index}/{len(endpoint_schedule)}] "
            f"weight={weight:.1f}, seed={seed}",
            flush=True,
        )
        embedding = interpolate_embeddings(
            embeddings["A"], embeddings["B"], weight
        )
        try:
            run = run_generation(
                model=model,
                embedding=embedding,
                output_dir=output_dir,
                weight=weight,
                seed=seed,
                canvas_length=args.canvas_length,
                steps=args.steps,
            )
        except Exception as error:
            runs.append(
                {
                    "weight": weight,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            results["status"] = "failed"
            write_results(results_path, results)
            raise
        runs.append(run)
        write_results(results_path, results)
        print(
            f"  completed in {run['duration_seconds']}s; "
            f"saved {run['output_file']}",
            flush=True,
        )

    endpoint_controls_passed = controls_pass(runs)
    results["endpoint_controls_passed"] = endpoint_controls_passed
    if not endpoint_controls_passed and not args.allow_failed_controls:
        results["status"] = "blocked_controls"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        print(
            "Endpoint controls failed; intermediate weights were not run.",
            flush=True,
        )
        return

    intermediate_schedule = [
        (weight, seed)
        for weight in WEIGHTS[1:-1]
        for seed in args.seeds
    ]
    for index, (weight, seed) in enumerate(intermediate_schedule, start=1):
        print(
            f"[sweep {index}/{len(intermediate_schedule)}] "
            f"weight={weight:.1f}, seed={seed}",
            flush=True,
        )
        embedding = interpolate_embeddings(
            embeddings["A"], embeddings["B"], weight
        )
        try:
            run = run_generation(
                model=model,
                embedding=embedding,
                output_dir=output_dir,
                weight=weight,
                seed=seed,
                canvas_length=args.canvas_length,
                steps=args.steps,
            )
        except Exception as error:
            runs.append(
                {
                    "weight": weight,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            results["status"] = "failed"
            write_results(results_path, results)
            raise
        runs.append(run)
        write_results(results_path, results)
        print(
            f"  completed in {run['duration_seconds']}s; "
            f"saved {run['output_file']}",
            flush=True,
        )

    results["status"] = "completed"
    results["completed_at"] = datetime.now(UTC).isoformat()
    results["candidate_hybrid_count"] = sum(
        bool(run["lexical_indicators"]["candidate_hybrid"])  # type: ignore[index]
        for run in runs
        if 0.0 < float(run["weight"]) < 1.0
    )
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
