from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

import torch

from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "011-nonsorting-endpoint-screen"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PAIRS_PATH = EXPERIMENT_DIR / "pairs.json"
EXPERIMENT_009_RUNNER = (
    REPOSITORY_ROOT
    / "experiments"
    / "009-multi-pair-replication"
    / "run.py"
)
DEFAULT_SEEDS = (42, 43, 44)


def load_experiment_009() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "experiment_009_runner",
        EXPERIMENT_009_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {EXPERIMENT_009_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E009 = load_experiment_009()


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
            "Run Experiment 011: screen four frozen non-sorting pairs for "
            "reliable A and B endpoints."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=96)
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
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def load_pair_document() -> tuple[str, list[dict[str, Any]]]:
    document = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    prompt = document.get("common_generation_prompt")
    pairs = document.get("pairs")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("pairs.json must contain a generation prompt")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs.json must contain a non-empty pair list")
    pair_ids = [pair.get("id") for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair IDs must be unique")
    return prompt, pairs


def endpoint_passes(
    runs: list[dict[str, Any]],
    *,
    weight: float,
    indicator: str,
    expected_seed_count: int,
) -> bool:
    matching = [
        run
        for run in runs
        if run["method"] == "dual_prompt_logit_mixing"
        and run["weight"] == weight
        and run["status"] == "completed"
    ]
    return len(matching) == expected_seed_count and all(
        run["denoised_lexical_indicators"][indicator]
        and not run["denoised_lexical_indicators"][
            "uses_prohibited_operation"
        ]
        for run in matching
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
    generation_prompt, pairs = load_pair_document()

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Pairs: {len(pairs)}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print("Loading the model once for all endpoint controls...", flush=True)
    model = DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model_id,
            prompt=generation_prompt,
            max_denoising_steps=args.steps,
            canvas_length=args.canvas_length,
            multi_canvas=1,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    )
    model.model._denoising_step = MethodType(
        E009.E008._dual_prompt_denoising_step,
        model.model,
    )

    results: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "repository_revision": git_revision(),
        "model": {
            "id": args.model_id,
            "revision": getattr(model.model.config, "_commit_hash", None),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "settings": {
            "canvas_length": args.canvas_length,
            "denoising_steps": args.steps,
            "seeds": list(args.seeds),
            "generation_prompt": generation_prompt,
            "conditions": ["A endpoint at weight 0.0", "B endpoint at weight 1.0"],
        },
        "inputs": {
            "pairs_path": str(PAIRS_PATH.relative_to(REPOSITORY_ROOT)),
            "pairs_sha256": file_sha256(PAIRS_PATH),
            "experiment_009_runner_path": str(
                EXPERIMENT_009_RUNNER.relative_to(REPOSITORY_ROOT)
            ),
            "experiment_009_runner_sha256": file_sha256(
                EXPERIMENT_009_RUNNER
            ),
        },
        "pair_results": [],
    }
    write_results(results_path, results)

    try:
        for pair_index, pair in enumerate(pairs, start=1):
            pair_id = pair["id"]
            print(
                f"[pair {pair_index}/{len(pairs)}] {pair_id}",
                flush=True,
            )
            pair_dir = output_dir / pair_id
            pair_dir.mkdir()
            prompts = (pair["source_prompt_a"], pair["source_prompt_b"])
            inputs = E009.E008.prompt_batch(model, prompts)
            token_count = int(inputs["input_ids"].shape[-1])
            expected_token_count = int(pair["prompt_token_count"])
            if token_count != expected_token_count:
                raise AssertionError(
                    f"{pair_id} token count changed: "
                    f"{token_count} != {expected_token_count}"
                )
            embeddings = {
                "A": model.text_to_embedding(
                    canvas,
                    prompt=prompts[0],
                    canvas_length=args.canvas_length,
                    multi_canvas=1,
                ),
                "B": model.text_to_embedding(
                    canvas,
                    prompt=prompts[1],
                    canvas_length=args.canvas_length,
                    multi_canvas=1,
                ),
            }
            weighted = {
                weight: E009.E008.interpolate_embeddings(
                    embeddings["A"],
                    embeddings["B"],
                    weight,
                )
                for weight in (0.0, 1.0)
            }
            pair_result: dict[str, Any] = {
                "pair_id": pair_id,
                "workload": pair["workload"],
                "expected_combined_concept": pair[
                    "expected_combined_concept"
                ],
                "status": "running",
                "prompt_alignment": {
                    "token_count": token_count,
                    "requires_padding": False,
                    "prompt_a_sha256": text_sha256(prompts[0]),
                    "prompt_b_sha256": text_sha256(prompts[1]),
                },
                "embedding_shapes": {
                    name: list(embedding.values.shape)
                    for name, embedding in embeddings.items()
                },
                "runs": [],
            }
            results["pair_results"].append(pair_result)
            write_results(results_path, results)

            for weight in (0.0, 1.0):
                for seed in args.seeds:
                    print(
                        f"  weight={weight:.1f}, seed={seed}",
                        flush=True,
                    )
                    run = E009.run_dual_condition(
                        model=model,
                        inputs=inputs,
                        embedding=weighted[weight],
                        pair=pair,
                        output_dir=pair_dir,
                        weight=weight,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                    pair_result["runs"].append(run)
                    write_results(results_path, results)

            pair_result["A_endpoint_passed"] = endpoint_passes(
                pair_result["runs"],
                weight=0.0,
                indicator="retains_A",
                expected_seed_count=len(args.seeds),
            )
            pair_result["B_endpoint_passed"] = endpoint_passes(
                pair_result["runs"],
                weight=1.0,
                indicator="retains_B",
                expected_seed_count=len(args.seeds),
            )
            pair_result["pair_endpoint_gate_passed"] = (
                pair_result["A_endpoint_passed"]
                and pair_result["B_endpoint_passed"]
            )
            pair_result["status"] = "completed"
            write_results(results_path, results)
    except Exception:
        results["status"] = "failed"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        raise

    passing_pair_ids = [
        pair["pair_id"]
        for pair in results["pair_results"]
        if pair["pair_endpoint_gate_passed"]
    ]
    results["summary"] = {
        "pair_count": len(results["pair_results"]),
        "passing_pair_count": len(passing_pair_ids),
        "passing_pair_ids": passing_pair_ids,
        "completed_run_count": sum(
            len(pair["runs"]) for pair in results["pair_results"]
        ),
    }
    results["status"] = (
        "completed_fixture"
        if args.model_id == LOCAL_TEST_MODEL_ID
        else "completed"
    )
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
