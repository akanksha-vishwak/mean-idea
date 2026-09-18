from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
    _generate_canvas,
)


EXPERIMENT_ID = "007-stochastic-projection"
REFERENCE_EXPERIMENT_ID = "006-top-candidate-projection"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
DEFAULT_REFERENCE_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / REFERENCE_EXPERIMENT_ID
    / "results"
    / "results.json"
)
GENERATION_PROMPT = (
    "Generate one concise optimization idea for the Python function my_sort. "
    "Preserve the specific algorithmic and implementation details represented "
    "in the supplied canvas. Do not use list.sort or sorted. Return only the "
    "optimization idea."
)
DEFAULT_TEMPERATURES = (0.5, 1.0, 2.0)
DEFAULT_ENDPOINT_SEEDS = (42, 43, 44)
DEFAULT_MIDPOINT_SEEDS = (42, 43, 44, 45, 46)
ENDPOINT_WEIGHTS = (0.0, 1.0)
MIDPOINT_WEIGHT = 0.5
DEFAULT_TOP_K = 20


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_positive_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated numbers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


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
            "Run Experiment 007: sample top-20 projection candidates at fixed "
            "temperatures, gate on endpoints, and test passing midpoints."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=64)
    parser.add_argument("--steps", type=positive_integer, default=8)
    parser.add_argument("--top-k", type=positive_integer, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--temperatures",
        type=parse_positive_floats,
        default=DEFAULT_TEMPERATURES,
        help="comma-separated positive temperatures",
    )
    parser.add_argument(
        "--endpoint-seeds",
        type=parse_seeds,
        default=DEFAULT_ENDPOINT_SEEDS,
        help="comma-separated endpoint-control seeds",
    )
    parser.add_argument(
        "--midpoint-seeds",
        type=parse_seeds,
        default=DEFAULT_MIDPOINT_SEEDS,
        help="comma-separated midpoint seeds",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=DEFAULT_REFERENCE_RESULTS_PATH,
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--allow-reference-model-mismatch",
        action="store_true",
        help="allow production candidate scores with a fixture model; testing only",
    )
    parser.add_argument(
        "--allow-failed-controls",
        action="store_true",
        help="run midpoint conditions despite endpoint failures; testing only",
    )
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
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lexical_indicators(text: str) -> dict[str, bool]:
    lowered = text.lower()
    counting = "counting sort" in lowered
    frequency = "frequen" in lowered or "count[" in lowered
    bounded = any(
        marker in lowered
        for marker in (
            "bounded",
            "10000",
            "9999",
            "known range",
            "fixed-size",
            "fixed size",
        )
    )
    numba = "numba" in lowered or "@njit" in lowered
    jit = "@njit" in lowered or "jit" in lowered or "machine code" in lowered
    prohibited = "list.sort" in lowered or "sorted(" in lowered
    retains_a = counting or (frequency and bounded)
    retains_b = numba and jit
    return {
        "mentions_counting_sort": counting,
        "mentions_frequency": frequency,
        "mentions_bounded_range": bounded,
        "retains_A": retains_a,
        "mentions_numba": numba,
        "mentions_jit_compilation": jit,
        "retains_B": retains_b,
        "retains_both": retains_a and retains_b,
        "uses_prohibited_builtin": prohibited,
        "candidate_hybrid": retains_a and retains_b and not prohibited,
    }


def load_reference_candidates(
    *,
    results_path: Path,
    model_id: str,
    canvas_length: int,
    top_k: int,
    allow_model_mismatch: bool,
) -> dict[str, Any]:
    manifest = json.loads(results_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != REFERENCE_EXPERIMENT_ID:
        raise ValueError(f"{results_path} is not an Experiment 006 manifest")
    if manifest.get("status") != "completed":
        raise ValueError(f"{results_path} is not completed")
    if (
        not allow_model_mismatch
        and manifest["model"]["id"] != model_id
    ):
        raise ValueError("reference model ID does not match this run")
    settings = manifest["settings"]
    if settings["canvas_length"] != canvas_length:
        raise ValueError("reference canvas length does not match this run")
    if settings["top_k"] < top_k:
        raise ValueError("reference candidate depth is smaller than requested top_k")
    if settings["generation_prompt"] != GENERATION_PROMPT:
        raise ValueError("reference generation prompt does not match this run")

    runs_by_weight = {
        float(run["weight"]): run
        for run in manifest["runs"]
        if float(run["weight"]) in (*ENDPOINT_WEIGHTS, MIDPOINT_WEIGHT)
    }
    required_weights = {*ENDPOINT_WEIGHTS, MIDPOINT_WEIGHT}
    if set(runs_by_weight) != required_weights:
        raise ValueError("reference manifest lacks endpoint or midpoint scores")

    candidates: dict[float, dict[str, Any]] = {}
    references = []
    for weight in sorted(required_weights):
        run = runs_by_weight[weight]
        candidate_path = results_path.parent / run["candidate_file"]
        if file_sha256(candidate_path) != run["candidate_file_sha256"]:
            raise ValueError(f"candidate checksum mismatch: {candidate_path}")
        details = json.loads(candidate_path.read_text(encoding="utf-8"))
        positions = sorted(details["positions"], key=lambda item: item["position"])
        if len(positions) != canvas_length:
            raise ValueError(f"{candidate_path} has an unexpected canvas length")
        token_ids = []
        scores = []
        for expected_position, position in enumerate(positions):
            if position["position"] != expected_position:
                raise ValueError(f"{candidate_path} has non-contiguous positions")
            top_candidates = position["candidates"][:top_k]
            if len(top_candidates) != top_k:
                raise ValueError(f"{candidate_path} lacks top-{top_k} candidates")
            token_ids.append([item["id"] for item in top_candidates])
            scores.append([item["score"] for item in top_candidates])
        candidates[weight] = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "scores": torch.tensor(scores, dtype=torch.float32),
        }
        references.append(
            {
                "weight": weight,
                "candidate_path": str(candidate_path.relative_to(REPOSITORY_ROOT)),
                "candidate_sha256": file_sha256(candidate_path),
            }
        )

    return {
        "manifest_path": str(results_path.relative_to(REPOSITORY_ROOT)),
        "manifest_sha256": file_sha256(results_path),
        "repository_revision": manifest["repository_revision"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "candidate_files": references,
        "candidates": candidates,
    }


def sample_top_k(
    *,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    if token_ids.shape != scores.shape:
        raise ValueError("token IDs and scores must have identical shapes")
    probabilities = torch.softmax(scores.float() / temperature, dim=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampled_indices = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
    sampled_ids = torch.gather(token_ids, 1, sampled_indices).squeeze(1)
    sampled_probabilities = torch.gather(
        probabilities,
        1,
        sampled_indices,
    ).squeeze(1)
    sampled_ranks = sampled_indices.squeeze(1) + 1
    entropy = -(
        probabilities
        * torch.log(probabilities.clamp_min(torch.finfo(torch.float32).tiny))
    ).sum(dim=1)
    return {
        "token_ids": sampled_ids,
        "ranks": sampled_ranks,
        "probabilities": sampled_probabilities,
        "mean_distribution_entropy": float(entropy.mean().item()),
    }


@torch.inference_mode()
def denoise_token_ids(
    *,
    model: DiffusionGemma,
    token_ids: torch.Tensor,
    canvas_length: int,
    steps: int,
) -> str:
    if token_ids.ndim != 1 or token_ids.numel() != canvas_length:
        raise ValueError("sampled token IDs must match the single canvas length")
    prompt_inputs = model._base_prompt_inputs(None)
    with model._using_canvas_settings(
        canvas_length,
        1,
        prompt_length=prompt_inputs["input_ids"].shape[-1],
    ) as (active_canvas_length, input_length):
        if input_length != canvas_length:
            raise ValueError("Experiment 007 supports exactly one canvas")
        context = torch.empty(
            (1, 0),
            dtype=torch.long,
            device=model.model.device,
        )
        inputs = model._prompt_inputs(
            prompt_inputs,
            context,
            torch.ones_like(context),
        )
        generated = _generate_canvas(
            model.model,
            inputs=inputs,
            decoder_input_ids=token_ids.unsqueeze(0).to(model.model.device),
            canvas_length=active_canvas_length,
            max_iterations=steps,
        )
    return model.processor.decode(
        generated[0],
        skip_special_tokens=True,
    ).strip()


def run_condition(
    *,
    model: DiffusionGemma,
    candidate_data: dict[str, Any],
    output_dir: Path,
    temperature: float,
    weight: float,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, Any]:
    sampled = sample_top_k(
        token_ids=candidate_data["token_ids"],
        scores=candidate_data["scores"],
        temperature=temperature,
        seed=seed,
    )
    initial_text = model.processor.decode(
        sampled["token_ids"],
        skip_special_tokens=True,
    ).strip()
    temperature_label = f"{temperature:g}".replace(".", "p")
    weight_label = f"{weight:.1f}".replace(".", "p")
    stem = f"temperature-{temperature_label}-weight-{weight_label}-seed-{seed}"
    initial_path = output_dir / f"{stem}-initial.txt"
    denoised_path = output_dir / f"{stem}-denoised.txt"
    initial_path.write_text(initial_text + "\n", encoding="utf-8")

    set_seed(seed)
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    denoised_text = denoise_token_ids(
        model=model,
        token_ids=sampled["token_ids"],
        canvas_length=canvas_length,
        steps=steps,
    )
    duration = time.monotonic() - started
    denoised_path.write_text(denoised_text + "\n", encoding="utf-8")

    ranks = sampled["ranks"]
    probabilities = sampled["probabilities"]
    return {
        "temperature": temperature,
        "weight": weight,
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "initial_output_file": initial_path.name,
        "initial_output_sha256": file_sha256(initial_path),
        "denoised_output_file": denoised_path.name,
        "denoised_output_sha256": file_sha256(denoised_path),
        "sampled_token_ids": sampled["token_ids"].tolist(),
        "sampled_candidate_ranks": ranks.tolist(),
        "sampling_summary": {
            "non_top_1_count": int((ranks != 1).sum().item()),
            "mean_selected_rank": float(ranks.float().mean().item()),
            "maximum_selected_rank": int(ranks.max().item()),
            "mean_selected_probability": float(
                probabilities.mean().item()
            ),
            "minimum_selected_probability": float(
                probabilities.min().item()
            ),
            "mean_distribution_entropy": sampled[
                "mean_distribution_entropy"
            ],
        },
        "initial_lexical_indicators": lexical_indicators(initial_text),
        "denoised_lexical_indicators": lexical_indicators(denoised_text),
    }


def temperature_controls_pass(
    runs: list[dict[str, Any]],
    temperature: float,
) -> bool:
    matching = [
        run
        for run in runs
        if run["temperature"] == temperature
        and run["weight"] in ENDPOINT_WEIGHTS
        and run["status"] == "completed"
    ]
    endpoint_a = [run for run in matching if run["weight"] == 0.0]
    endpoint_b = [run for run in matching if run["weight"] == 1.0]
    return bool(endpoint_a and endpoint_b) and all(
        run["denoised_lexical_indicators"]["retains_A"]
        for run in endpoint_a
    ) and all(
        run["denoised_lexical_indicators"]["retains_B"]
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

    reference = load_reference_candidates(
        results_path=args.reference_results.resolve(),
        model_id=args.model_id,
        canvas_length=args.canvas_length,
        top_k=args.top_k,
        allow_model_mismatch=args.allow_reference_model_mismatch,
    )

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Temperatures: {args.temperatures}", flush=True)
    print(f"Endpoint seeds: {args.endpoint_seeds}", flush=True)
    print(f"Midpoint seeds: {args.midpoint_seeds}", flush=True)
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
    model_revision = getattr(model.model.config, "_commit_hash", None)
    if (
        not args.allow_reference_model_mismatch
        and reference["model_revision"] is not None
        and model_revision != reference["model_revision"]
    ):
        raise ValueError("loaded model revision does not match Experiment 006")

    runs: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "repository_revision": git_revision(),
        "model": {
            "id": args.model_id,
            "revision": model_revision,
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
            "top_k": args.top_k,
            "temperatures": list(args.temperatures),
            "endpoint_seeds": list(args.endpoint_seeds),
            "midpoint_seeds": list(args.midpoint_seeds),
            "sampling": "independent categorical sampling from temperature-scaled top-k scores",
            "generation_prompt": GENERATION_PROMPT,
            "allow_reference_model_mismatch": args.allow_reference_model_mismatch,
            "allow_failed_controls": args.allow_failed_controls,
        },
        "reference": {
            key: value
            for key, value in reference.items()
            if key != "candidates"
        },
        "runs": runs,
    }
    write_results(results_path, results)

    endpoint_schedule = [
        (temperature, weight, seed)
        for temperature in args.temperatures
        for weight in ENDPOINT_WEIGHTS
        for seed in args.endpoint_seeds
    ]
    try:
        for index, (temperature, weight, seed) in enumerate(
            endpoint_schedule,
            start=1,
        ):
            print(
                f"[control {index}/{len(endpoint_schedule)}] "
                f"temperature={temperature:g}, weight={weight:.1f}, seed={seed}",
                flush=True,
            )
            try:
                run = run_condition(
                    model=model,
                    candidate_data=reference["candidates"][weight],
                    output_dir=output_dir,
                    temperature=temperature,
                    weight=weight,
                    seed=seed,
                    canvas_length=args.canvas_length,
                    steps=args.steps,
                )
            except Exception as error:
                runs.append(
                    {
                        "temperature": temperature,
                        "weight": weight,
                        "seed": seed,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                write_results(results_path, results)
                raise
            runs.append(run)
            write_results(results_path, results)
            print(
                f"  completed in {run['duration_seconds']}s; "
                f"saved {run['denoised_output_file']}",
                flush=True,
            )

        qualified_temperatures = [
            temperature
            for temperature in args.temperatures
            if temperature_controls_pass(runs, temperature)
        ]
        results["qualified_temperatures"] = qualified_temperatures
        results["temperature_control_results"] = {
            str(temperature): temperature in qualified_temperatures
            for temperature in args.temperatures
        }
        write_results(results_path, results)

        midpoint_temperatures = (
            list(args.temperatures)
            if args.allow_failed_controls
            else qualified_temperatures
        )
        if not midpoint_temperatures:
            results["status"] = "blocked_controls"
            results["completed_at"] = datetime.now(UTC).isoformat()
            write_results(results_path, results)
            print(
                "No temperature passed both endpoint gates; midpoint runs skipped.",
                flush=True,
            )
            return

        midpoint_schedule = [
            (temperature, seed)
            for temperature in midpoint_temperatures
            for seed in args.midpoint_seeds
        ]
        for index, (temperature, seed) in enumerate(
            midpoint_schedule,
            start=1,
        ):
            print(
                f"[midpoint {index}/{len(midpoint_schedule)}] "
                f"temperature={temperature:g}, seed={seed}",
                flush=True,
            )
            try:
                run = run_condition(
                    model=model,
                    candidate_data=reference["candidates"][MIDPOINT_WEIGHT],
                    output_dir=output_dir,
                    temperature=temperature,
                    weight=MIDPOINT_WEIGHT,
                    seed=seed,
                    canvas_length=args.canvas_length,
                    steps=args.steps,
                )
            except Exception as error:
                runs.append(
                    {
                        "temperature": temperature,
                        "weight": MIDPOINT_WEIGHT,
                        "seed": seed,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                write_results(results_path, results)
                raise
            runs.append(run)
            write_results(results_path, results)
            print(
                f"  completed in {run['duration_seconds']}s; "
                f"saved {run['denoised_output_file']}",
                flush=True,
            )
    except Exception as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        write_results(results_path, results)
        raise

    results["candidate_hybrid_count"] = sum(
        run["denoised_lexical_indicators"]["candidate_hybrid"]
        for run in runs
        if run["weight"] == MIDPOINT_WEIGHT
    )
    results["status"] = (
        "completed_fixture"
        if args.allow_reference_model_mismatch
        or args.allow_failed_controls
        or args.model_id == LOCAL_TEST_MODEL_ID
        else "completed"
    )
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
