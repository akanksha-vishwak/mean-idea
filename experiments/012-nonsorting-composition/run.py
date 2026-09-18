from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
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


EXPERIMENT_ID = "012-nonsorting-composition"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
EXPERIMENT_011_PAIRS = (
    REPOSITORY_ROOT
    / "experiments"
    / "011-nonsorting-endpoint-screen"
    / "pairs.json"
)
EXPERIMENT_011_RESULTS = (
    REPOSITORY_ROOT
    / "experiments"
    / "011-nonsorting-endpoint-screen"
    / "results"
    / "results.json"
)
EXPERIMENT_009_RUNNER = (
    REPOSITORY_ROOT
    / "experiments"
    / "009-multi-pair-replication"
    / "run.py"
)
DIRECT_PROMPTS_PATH = EXPERIMENT_DIR / "direct-prompts.json"
EXPECTED_PAIR_IDS = (
    "matrix-blocked-blas",
    "graph-csr-deque-bfs",
    "image-batched-vectorization",
)
DEFAULT_SEEDS = (42, 43, 44)
MIDPOINT_WEIGHT = 0.5


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
    seeds = tuple(int(item.strip()) for item in value.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final endpoint-gated non-sorting composition test."
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=96)
    parser.add_argument("--steps", type=positive_integer, default=8)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
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
    return completed.stdout.strip() if completed.returncode == 0 else None


def load_inputs() -> tuple[str, list[dict[str, Any]], dict[str, str], dict]:
    pair_document = json.loads(
        EXPERIMENT_011_PAIRS.read_text(encoding="utf-8")
    )
    reference = json.loads(
        EXPERIMENT_011_RESULTS.read_text(encoding="utf-8")
    )
    direct_prompts = json.loads(
        DIRECT_PROMPTS_PATH.read_text(encoding="utf-8")
    )["prompts"]
    passing_ids = tuple(reference["summary"]["passing_pair_ids"])
    if passing_ids != EXPECTED_PAIR_IDS:
        raise ValueError(
            f"unexpected passing pair IDs: {passing_ids} != {EXPECTED_PAIR_IDS}"
        )
    pairs_by_id = {pair["id"]: pair for pair in pair_document["pairs"]}
    pairs = [pairs_by_id[pair_id] for pair_id in EXPECTED_PAIR_IDS]
    if set(direct_prompts) != set(EXPECTED_PAIR_IDS):
        raise ValueError("direct prompts do not match the selected pair IDs")
    return (
        pair_document["common_generation_prompt"],
        pairs,
        direct_prompts,
        reference,
    )


def generate_direct(
    *,
    model: DiffusionGemma,
    prompt: str,
    seed: int,
    canvas_length: int,
    steps: int,
) -> str:
    E009.E008.set_seed(seed)
    inputs = model._base_prompt_inputs(prompt)
    output = model.model.generate(
        **inputs,
        max_new_tokens=canvas_length,
        max_denoising_steps=steps,
        disable_compile=True,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    prompt_length = inputs["input_ids"].shape[-1]
    generated = sequences[0, prompt_length : prompt_length + canvas_length]
    return model.processor.decode(generated, skip_special_tokens=True).strip()


def run_direct(
    *,
    model: DiffusionGemma,
    pair: dict[str, Any],
    prompt: str,
    output_dir: Path,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, Any]:
    output_path = output_dir / f"direct-prompt-seed-{seed}.txt"
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    text = generate_direct(
        model=model,
        prompt=prompt,
        seed=seed,
        canvas_length=canvas_length,
        steps=steps,
    )
    duration = time.monotonic() - started
    output_path.write_text(text + "\n", encoding="utf-8")
    return {
        "method": "direct_diffusion_prompt",
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "output_file": output_path.name,
        "output_sha256": file_sha256(output_path),
        "lexical_indicators": E009.lexical_indicators(text, pair),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    results_path = output_dir / "results.json"
    if results_path.exists():
        raise FileExistsError(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = BLANK_CANVAS_PATH.read_text(encoding="utf-8")
    if canvas:
        raise ValueError("blank canvas must be empty")
    generation_prompt, pairs, direct_prompts, reference = load_inputs()

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Pairs: {len(pairs)}; seeds: {args.seeds}", flush=True)
    print("Loading model once for 27 conditions...", flush=True)
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
    reference_model_revision = reference["model"]["revision"]
    current_model_revision = getattr(model.model.config, "_commit_hash", None)
    if (
        args.model_id != LOCAL_TEST_MODEL_ID
        and current_model_revision != reference_model_revision
    ):
        raise ValueError("model revision differs from Experiment 011")

    results: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "repository_revision": git_revision(),
        "model": {
            "id": args.model_id,
            "revision": current_model_revision,
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
            "weight": MIDPOINT_WEIGHT,
            "methods": [
                "dual_prompt_logit_mixing",
                "aligned_hard_projection",
                "direct_diffusion_prompt",
            ],
        },
        "reference": {
            "endpoint_manifest_path": str(
                EXPERIMENT_011_RESULTS.relative_to(REPOSITORY_ROOT)
            ),
            "endpoint_manifest_sha256": file_sha256(EXPERIMENT_011_RESULTS),
            "passing_pair_ids": list(EXPECTED_PAIR_IDS),
            "model_revision": reference_model_revision,
        },
        "inputs": {
            "pairs_path": str(
                EXPERIMENT_011_PAIRS.relative_to(REPOSITORY_ROOT)
            ),
            "pairs_sha256": file_sha256(EXPERIMENT_011_PAIRS),
            "direct_prompts_path": str(
                DIRECT_PROMPTS_PATH.relative_to(REPOSITORY_ROOT)
            ),
            "direct_prompts_sha256": file_sha256(DIRECT_PROMPTS_PATH),
        },
        "pair_results": [],
    }
    write_results(results_path, results)

    try:
        for index, pair in enumerate(pairs, start=1):
            pair_id = pair["id"]
            print(f"[pair {index}/{len(pairs)}] {pair_id}", flush=True)
            pair_dir = output_dir / pair_id
            pair_dir.mkdir()
            prompts = (pair["source_prompt_a"], pair["source_prompt_b"])
            inputs = E009.E008.prompt_batch(model, prompts)
            token_count = int(inputs["input_ids"].shape[-1])
            if token_count != pair["prompt_token_count"]:
                raise AssertionError(f"{pair_id} prompt length changed")
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
            midpoint = E009.E008.interpolate_embeddings(
                embeddings["A"],
                embeddings["B"],
                MIDPOINT_WEIGHT,
            )
            pair_result: dict[str, Any] = {
                "pair_id": pair_id,
                "workload": pair["workload"],
                "expected_combined_concept": pair[
                    "expected_combined_concept"
                ],
                "prompt_token_count": token_count,
                "runs": [],
            }
            results["pair_results"].append(pair_result)
            write_results(results_path, results)

            for seed in args.seeds:
                print(f"  [dual midpoint] seed={seed}", flush=True)
                pair_result["runs"].append(
                    E009.run_dual_condition(
                        model=model,
                        inputs=inputs,
                        embedding=midpoint,
                        pair=pair,
                        output_dir=pair_dir,
                        weight=MIDPOINT_WEIGHT,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                )
                write_results(results_path, results)
            for seed in args.seeds:
                print(f"  [hard midpoint] seed={seed}", flush=True)
                pair_result["runs"].append(
                    E009.run_hard_projection_condition(
                        model=model,
                        embedding=midpoint,
                        pair=pair,
                        output_dir=pair_dir,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                )
                write_results(results_path, results)
            for seed in args.seeds:
                print(f"  [direct prompt] seed={seed}", flush=True)
                pair_result["runs"].append(
                    run_direct(
                        model=model,
                        pair=pair,
                        prompt=direct_prompts[pair_id],
                        output_dir=pair_dir,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                )
                write_results(results_path, results)

            pair_result["summary"] = {
                "dual_midpoint_hybrid_count": sum(
                    run["denoised_lexical_indicators"]["candidate_hybrid"]
                    for run in pair_result["runs"]
                    if run["method"] == "dual_prompt_logit_mixing"
                ),
                "hard_midpoint_hybrid_count": sum(
                    run["denoised_lexical_indicators"]["candidate_hybrid"]
                    for run in pair_result["runs"]
                    if run["method"] == "aligned_hard_projection"
                ),
                "direct_prompt_hybrid_count": sum(
                    run["lexical_indicators"]["candidate_hybrid"]
                    for run in pair_result["runs"]
                    if run["method"] == "direct_diffusion_prompt"
                ),
            }
            write_results(results_path, results)
    except Exception:
        results["status"] = "failed"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        raise

    results["summary"] = {
        "pair_count": len(results["pair_results"]),
        "completed_run_count": sum(
            len(pair["runs"]) for pair in results["pair_results"]
        ),
        "dual_midpoint_hybrid_count": sum(
            pair["summary"]["dual_midpoint_hybrid_count"]
            for pair in results["pair_results"]
        ),
        "hard_midpoint_hybrid_count": sum(
            pair["summary"]["hard_midpoint_hybrid_count"]
            for pair in results["pair_results"]
        ),
        "direct_prompt_hybrid_count": sum(
            pair["summary"]["direct_prompt_hybrid_count"]
            for pair in results["pair_results"]
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
