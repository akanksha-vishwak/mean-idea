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

from mean_idea.api import TextEmbedding
from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "009-multi-pair-replication"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PAIRS_PATH = EXPERIMENT_DIR / "pairs.json"
EXPERIMENT_008_RUNNER = (
    REPOSITORY_ROOT
    / "experiments"
    / "008-dual-prompt-logit-mixing"
    / "run.py"
)
GENERATION_PROMPT = (
    "Generate one concise optimization idea for the stated Python task. "
    "Preserve the specific algorithmic and implementation details represented "
    "in the supplied canvas. Return only the optimization idea."
)
DEFAULT_SEEDS = (42, 43, 44)
MIDPOINT_WEIGHT = 0.5


def load_experiment_008() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "experiment_008_runner",
        EXPERIMENT_008_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {EXPERIMENT_008_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E008 = load_experiment_008()


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
            "Run Experiment 009: replicate midpoint mechanisms and direct "
            "prompting across three frozen idea pairs."
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
    parser.add_argument(
        "--allow-failed-controls",
        action="store_true",
        help="continue after endpoint failure; intended only for fixture testing",
    )
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


def load_pairs() -> list[dict[str, Any]]:
    document = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs.json must contain a non-empty pairs list")
    pair_ids = [pair.get("id") for pair in pairs]
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise ValueError("every pair must have a non-empty string ID")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair IDs must be unique")
    return pairs


def lexical_indicators(text: str, pair: dict[str, Any]) -> dict[str, Any]:
    lowered = text.lower()

    def retains(groups: list[list[str]]) -> bool:
        return all(
            any(marker.lower() in lowered for marker in group)
            for group in groups
        )

    matched_a = [
        [marker for marker in group if marker.lower() in lowered]
        for group in pair["concept_a_groups"]
    ]
    matched_b = [
        [marker for marker in group if marker.lower() in lowered]
        for group in pair["concept_b_groups"]
    ]
    prohibited = [
        marker
        for marker in pair["prohibited_markers"]
        if marker.lower() in lowered
    ]
    retains_a = retains(pair["concept_a_groups"])
    retains_b = retains(pair["concept_b_groups"])
    return {
        "retains_A": retains_a,
        "retains_B": retains_b,
        "retains_both": retains_a and retains_b,
        "uses_prohibited_operation": bool(prohibited),
        "candidate_hybrid": retains_a and retains_b and not prohibited,
        "matched_A_groups": matched_a,
        "matched_B_groups": matched_b,
        "matched_prohibited_markers": prohibited,
    }


def generate_direct_prompt(
    *,
    model: DiffusionGemma,
    prompt: str,
    seed: int,
    canvas_length: int,
    steps: int,
) -> str:
    E008.set_seed(seed)
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
    return model.processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def run_dual_condition(
    *,
    model: DiffusionGemma,
    inputs,
    embedding: TextEmbedding,
    pair: dict[str, Any],
    output_dir: Path,
    weight: float,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, Any]:
    initial_ids = model._project_to_tokens(embedding.values).cpu()
    initial_text = model.processor.decode(
        initial_ids,
        skip_special_tokens=True,
    ).strip()
    weight_label = str(weight).replace(".", "p")
    stem = f"dual-weight-{weight_label}-seed-{seed}"
    initial_path = output_dir / f"{stem}-initial.txt"
    output_path = output_dir / f"{stem}-denoised.txt"
    trace_path = output_dir / f"{stem}-trace.json"
    initial_path.write_text(initial_text + "\n", encoding="utf-8")
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    text, trace = E008.generate_dual_prompt(
        model=model,
        inputs=inputs,
        initial_token_ids=initial_ids,
        weight=weight,
        seed=seed,
        canvas_length=canvas_length,
        steps=steps,
    )
    duration = time.monotonic() - started
    output_path.write_text(text + "\n", encoding="utf-8")
    write_results(trace_path, {"weight": weight, "seed": seed, "steps": trace})
    return {
        "method": "dual_prompt_logit_mixing",
        "weight": weight,
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "initial_output_file": initial_path.name,
        "initial_output_sha256": file_sha256(initial_path),
        "denoised_output_file": output_path.name,
        "denoised_output_sha256": file_sha256(output_path),
        "trace_file": trace_path.name,
        "trace_sha256": file_sha256(trace_path),
        "initial_lexical_indicators": lexical_indicators(initial_text, pair),
        "denoised_lexical_indicators": lexical_indicators(text, pair),
    }


def run_hard_projection_condition(
    *,
    model: DiffusionGemma,
    embedding: TextEmbedding,
    pair: dict[str, Any],
    output_dir: Path,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, Any]:
    initial_ids = model._project_to_tokens(embedding.values).cpu()
    initial_text = model.processor.decode(
        initial_ids,
        skip_special_tokens=True,
    ).strip()
    stem = f"hard-midpoint-seed-{seed}"
    initial_path = output_dir / f"{stem}-initial.txt"
    output_path = output_dir / f"{stem}-denoised.txt"
    initial_path.write_text(initial_text + "\n", encoding="utf-8")
    E008.set_seed(seed)
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
        "method": "aligned_hard_projection",
        "weight": MIDPOINT_WEIGHT,
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "initial_output_file": initial_path.name,
        "initial_output_sha256": file_sha256(initial_path),
        "denoised_output_file": output_path.name,
        "denoised_output_sha256": file_sha256(output_path),
        "initial_lexical_indicators": lexical_indicators(initial_text, pair),
        "denoised_lexical_indicators": lexical_indicators(text, pair),
    }


def run_direct_condition(
    *,
    model: DiffusionGemma,
    pair: dict[str, Any],
    output_dir: Path,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, Any]:
    output_path = output_dir / f"direct-prompt-seed-{seed}.txt"
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    text = generate_direct_prompt(
        model=model,
        prompt=pair["direct_prompt"],
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
        "lexical_indicators": lexical_indicators(text, pair),
    }


def endpoint_controls_pass(
    runs: list[dict[str, Any]],
    expected_seed_count: int,
) -> bool:
    endpoint_a = [
        run
        for run in runs
        if run["method"] == "dual_prompt_logit_mixing"
        and run["weight"] == 0.0
        and run["status"] == "completed"
    ]
    endpoint_b = [
        run
        for run in runs
        if run["method"] == "dual_prompt_logit_mixing"
        and run["weight"] == 1.0
        and run["status"] == "completed"
    ]
    return (
        len(endpoint_a) == expected_seed_count
        and len(endpoint_b) == expected_seed_count
        and all(
            run["denoised_lexical_indicators"]["retains_A"]
            for run in endpoint_a
        )
        and all(
            run["denoised_lexical_indicators"]["retains_B"]
            for run in endpoint_b
        )
    )


def append_failed_run(
    pair_result: dict[str, Any],
    *,
    method: str,
    seed: int,
    error: Exception,
    weight: float | None = None,
) -> None:
    run = {
        "method": method,
        "seed": seed,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }
    if weight is not None:
        run["weight"] = weight
    pair_result["runs"].append(run)


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
    pairs = load_pairs()

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Pairs: {len(pairs)}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print("Loading the model once for all pairs...", flush=True)
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
    model.model._denoising_step = MethodType(
        E008._dual_prompt_denoising_step,
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
            "multi_canvas": 1,
            "denoising_steps": args.steps,
            "weights": [0.0, MIDPOINT_WEIGHT, 1.0],
            "seeds": list(args.seeds),
            "generation_prompt": GENERATION_PROMPT,
            "endpoint_gate": "all A and B seeds retain their source concepts",
            "midpoint_methods": [
                "aligned hard projection",
                "per-step linear output-logit mixing",
            ],
        },
        "canvas": {
            "path": str(BLANK_CANVAS_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": file_sha256(BLANK_CANVAS_PATH),
            "character_count": len(canvas),
        },
        "inputs": {
            "pairs_path": str(PAIRS_PATH.relative_to(REPOSITORY_ROOT)),
            "pairs_sha256": file_sha256(PAIRS_PATH),
            "experiment_008_runner_path": str(
                EXPERIMENT_008_RUNNER.relative_to(REPOSITORY_ROOT)
            ),
            "experiment_008_runner_sha256": file_sha256(
                EXPERIMENT_008_RUNNER
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
            dual_inputs = E008.prompt_batch(model, prompts)
            prompt_token_count = int(dual_inputs["input_ids"].shape[-1])
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
            weighted_embeddings = {
                weight: E008.interpolate_embeddings(
                    embeddings["A"],
                    embeddings["B"],
                    weight,
                )
                for weight in (0.0, MIDPOINT_WEIGHT, 1.0)
            }
            midpoint = weighted_embeddings[MIDPOINT_WEIGHT]
            pair_result: dict[str, Any] = {
                "pair_id": pair_id,
                "workload": pair["workload"],
                "provenance": pair["provenance"],
                "expected_combined_concept": pair[
                    "expected_combined_concept"
                ],
                "status": "running",
                "prompt_alignment": {
                    "token_count": prompt_token_count,
                    "requires_padding": False,
                    "prompt_a_sha256": text_sha256(prompts[0]),
                    "prompt_b_sha256": text_sha256(prompts[1]),
                    "direct_prompt_sha256": text_sha256(
                        pair["direct_prompt"]
                    ),
                },
                "embedding_shapes": {
                    name: list(embedding.values.shape)
                    for name, embedding in embeddings.items()
                },
                "runs": [],
            }
            results["pair_results"].append(pair_result)
            write_results(results_path, results)

            endpoint_schedule = [
                (weight, seed)
                for weight in (0.0, 1.0)
                for seed in args.seeds
            ]
            for control_index, (weight, seed) in enumerate(
                endpoint_schedule,
                start=1,
            ):
                print(
                    f"  [control {control_index}/{len(endpoint_schedule)}] "
                    f"weight={weight:.1f}, seed={seed}",
                    flush=True,
                )
                try:
                    run = run_dual_condition(
                        model=model,
                        inputs=dual_inputs,
                        embedding=weighted_embeddings[weight],
                        pair=pair,
                        output_dir=pair_dir,
                        weight=weight,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                except Exception as error:
                    append_failed_run(
                        pair_result,
                        method="dual_prompt_logit_mixing",
                        weight=weight,
                        seed=seed,
                        error=error,
                    )
                    write_results(results_path, results)
                    raise
                pair_result["runs"].append(run)
                write_results(results_path, results)

            controls_passed = endpoint_controls_pass(
                pair_result["runs"],
                len(args.seeds),
            )
            pair_result["endpoint_controls_passed"] = controls_passed
            write_results(results_path, results)
            if not controls_passed and not args.allow_failed_controls:
                pair_result["status"] = "blocked_controls"
                write_results(results_path, results)
                print(
                    "  endpoint gate failed; midpoint and direct runs skipped",
                    flush=True,
                )
                continue

            for seed in args.seeds:
                print(
                    f"  [dual midpoint] seed={seed}",
                    flush=True,
                )
                try:
                    run = run_dual_condition(
                        model=model,
                        inputs=dual_inputs,
                        embedding=midpoint,
                        pair=pair,
                        output_dir=pair_dir,
                        weight=MIDPOINT_WEIGHT,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                except Exception as error:
                    append_failed_run(
                        pair_result,
                        method="dual_prompt_logit_mixing",
                        weight=MIDPOINT_WEIGHT,
                        seed=seed,
                        error=error,
                    )
                    write_results(results_path, results)
                    raise
                pair_result["runs"].append(run)
                write_results(results_path, results)

            for seed in args.seeds:
                print(
                    f"  [hard midpoint] seed={seed}",
                    flush=True,
                )
                try:
                    run = run_hard_projection_condition(
                        model=model,
                        embedding=midpoint,
                        pair=pair,
                        output_dir=pair_dir,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                except Exception as error:
                    append_failed_run(
                        pair_result,
                        method="aligned_hard_projection",
                        weight=MIDPOINT_WEIGHT,
                        seed=seed,
                        error=error,
                    )
                    write_results(results_path, results)
                    raise
                pair_result["runs"].append(run)
                write_results(results_path, results)

            for seed in args.seeds:
                print(
                    f"  [direct prompt] seed={seed}",
                    flush=True,
                )
                try:
                    run = run_direct_condition(
                        model=model,
                        pair=pair,
                        output_dir=pair_dir,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                except Exception as error:
                    append_failed_run(
                        pair_result,
                        method="direct_diffusion_prompt",
                        seed=seed,
                        error=error,
                    )
                    write_results(results_path, results)
                    raise
                pair_result["runs"].append(run)
                write_results(results_path, results)

            pair_result["summary"] = {
                "dual_midpoint_hybrid_count": sum(
                    run["denoised_lexical_indicators"]["candidate_hybrid"]
                    for run in pair_result["runs"]
                    if run["method"] == "dual_prompt_logit_mixing"
                    and run["weight"] == MIDPOINT_WEIGHT
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
            pair_result["status"] = (
                "completed_fixture"
                if args.allow_failed_controls
                or args.model_id == LOCAL_TEST_MODEL_ID
                else "completed"
            )
            write_results(results_path, results)
    except Exception:
        results["status"] = "failed"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        raise

    interpretable_pairs = [
        pair
        for pair in results["pair_results"]
        if pair.get("endpoint_controls_passed")
    ]
    results["summary"] = {
        "pair_count": len(results["pair_results"]),
        "interpretable_pair_count": len(interpretable_pairs),
        "blocked_pair_count": sum(
            pair["status"] == "blocked_controls"
            for pair in results["pair_results"]
        ),
        "dual_midpoint_hybrid_count": sum(
            pair.get("summary", {}).get("dual_midpoint_hybrid_count", 0)
            for pair in results["pair_results"]
        ),
        "hard_midpoint_hybrid_count": sum(
            pair.get("summary", {}).get("hard_midpoint_hybrid_count", 0)
            for pair in results["pair_results"]
        ),
        "direct_prompt_hybrid_count": sum(
            pair.get("summary", {}).get("direct_prompt_hybrid_count", 0)
            for pair in results["pair_results"]
        ),
        "completed_run_count": sum(
            run["status"] == "completed"
            for pair in results["pair_results"]
            for run in pair["runs"]
        ),
    }
    if args.allow_failed_controls or args.model_id == LOCAL_TEST_MODEL_ID:
        results["status"] = "completed_fixture"
    elif len(interpretable_pairs) == len(pairs):
        results["status"] = "completed"
    else:
        results["status"] = "completed_with_blocked_pairs"
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
