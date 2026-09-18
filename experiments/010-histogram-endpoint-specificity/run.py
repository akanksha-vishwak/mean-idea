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


EXPERIMENT_ID = "010-histogram-endpoint-specificity"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PROMPT_A_PATH = EXPERIMENT_DIR / "prompt-a-strengthened.txt"
PROMPT_B_PATH = EXPERIMENT_DIR / "prompt-b-strengthened.txt"
EXPERIMENT_009_RUNNER = (
    REPOSITORY_ROOT
    / "experiments"
    / "009-multi-pair-replication"
    / "run.py"
)
EXPERIMENT_009_PAIRS = (
    REPOSITORY_ROOT
    / "experiments"
    / "009-multi-pair-replication"
    / "pairs.json"
)
REFERENCE_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "experiments"
    / "009-multi-pair-replication"
    / "results"
    / "pair-parallel-byte-histogram"
)
DEFAULT_SEEDS = (42, 43, 44)
ORIGINAL_PROMPT_TOKEN_COUNT = 126
STRENGTHENED_PROMPT_TOKEN_COUNT = 153


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
            "Run Experiment 010: reproduce and strengthen the blocked "
            "byte-histogram multiprocessing endpoint."
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


def canonical_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def canonical_text_sha256(value: str) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


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


def load_original_pair() -> dict[str, Any]:
    document = json.loads(EXPERIMENT_009_PAIRS.read_text(encoding="utf-8"))
    matches = [
        pair
        for pair in document["pairs"]
        if pair["id"] == "pair-parallel-byte-histogram"
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one frozen byte-histogram pair")
    return matches[0]


def strengthened_pair(prompts: tuple[str, str]) -> dict[str, Any]:
    return {
        "id": "pair-parallel-byte-histogram-strengthened",
        "workload": "byte-frequency-histogram",
        "source_prompt_a": prompts[0],
        "source_prompt_b": prompts[1],
        "concept_a_groups": [
            ["numpy", "np."],
            ["frombuffer"],
            ["bincount"],
        ],
        "concept_b_groups": [
            ["multiprocessing", "pool.map", "process pool"],
            ["chunk"],
            ["private", "local", "partial"],
            ["histogram"],
            ["parent", "main process"],
            ["sum", "combine", "aggregate", "reduce"],
        ],
        "prohibited_markers": [
            "multiprocessing.array",
            "shared memory",
            "shared buffer",
            "shared state",
        ],
    }


def proposes_shared_state(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "multiprocessing.array",
        "shared memory",
        "shared buffer",
        "shared state",
    )
    negations = (
        "avoid ",
        "do not use ",
        "don't use ",
        "without ",
        "rather than ",
        "instead of ",
        "no ",
    )
    for marker in markers:
        start = 0
        while True:
            index = lowered.find(marker, start)
            if index < 0:
                break
            prefix = lowered[max(0, index - 40) : index]
            if not any(negation in prefix for negation in negations):
                return True
            start = index + len(marker)
    return False


def refresh_strategy_indicators(
    run: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    for file_key, indicator_key in (
        ("initial_output_file", "initial_lexical_indicators"),
        ("denoised_output_file", "denoised_lexical_indicators"),
    ):
        text = (output_dir / run[file_key]).read_text(encoding="utf-8")
        indicators = run[indicator_key]
        prohibited = proposes_shared_state(text)
        indicators["uses_prohibited_operation"] = prohibited
        indicators["matched_prohibited_markers"] = (
            ["positive shared-state strategy"] if prohibited else []
        )
        indicators["candidate_hybrid"] = (
            indicators["retains_A"]
            and indicators["retains_B"]
            and not prohibited
        )


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


def run_endpoint_set(
    *,
    model: DiffusionGemma,
    canvas: str,
    pair: dict[str, Any],
    weights: tuple[float, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    canvas_length: int,
    steps: int,
) -> tuple[list[dict[str, Any]], int]:
    prompts = (pair["source_prompt_a"], pair["source_prompt_b"])
    inputs = E009.E008.prompt_batch(model, prompts)
    token_count = int(inputs["input_ids"].shape[-1])
    embeddings = {
        "A": model.text_to_embedding(
            canvas,
            prompt=prompts[0],
            canvas_length=canvas_length,
            multi_canvas=1,
        ),
        "B": model.text_to_embedding(
            canvas,
            prompt=prompts[1],
            canvas_length=canvas_length,
            multi_canvas=1,
        ),
    }
    weighted = {
        weight: E009.E008.interpolate_embeddings(
            embeddings["A"],
            embeddings["B"],
            weight,
        )
        for weight in weights
    }
    runs: list[dict[str, Any]] = []
    for weight in weights:
        for seed in seeds:
            print(
                f"  weight={weight:.1f}, seed={seed}",
                flush=True,
            )
            run = E009.run_dual_condition(
                model=model,
                inputs=inputs,
                embedding=weighted[weight],
                pair=pair,
                output_dir=output_dir,
                weight=weight,
                seed=seed,
                canvas_length=canvas_length,
                steps=steps,
            )
            refresh_strategy_indicators(run, output_dir=output_dir)
            runs.append(run)
    return runs, token_count


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    results_path = output_dir / "results.json"
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = output_dir / "original-b-reproduction"
    strengthened_dir = output_dir / "strengthened-aligned"
    original_dir.mkdir()
    strengthened_dir.mkdir()

    canvas = BLANK_CANVAS_PATH.read_text(encoding="utf-8")
    if canvas:
        raise ValueError(f"{BLANK_CANVAS_PATH} must be empty")
    original_pair = load_original_pair()
    prompts = (
        PROMPT_A_PATH.read_text(encoding="utf-8"),
        PROMPT_B_PATH.read_text(encoding="utf-8"),
    )
    explicit_pair = strengthened_pair(prompts)

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print("Loading the model once for all endpoint conditions...", flush=True)
    model = DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model_id,
            prompt=E009.GENERATION_PROMPT,
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
            "generation_prompt": E009.GENERATION_PROMPT,
            "conditions": [
                "original B reproduction",
                "strengthened A endpoint",
                "strengthened B endpoint",
            ],
        },
        "inputs": {
            "original_pairs_path": str(
                EXPERIMENT_009_PAIRS.relative_to(REPOSITORY_ROOT)
            ),
            "original_pairs_sha256": file_sha256(EXPERIMENT_009_PAIRS),
            "reference_output_dir": str(
                REFERENCE_OUTPUT_DIR.relative_to(REPOSITORY_ROOT)
            ),
            "prompt_a_path": str(PROMPT_A_PATH.relative_to(REPOSITORY_ROOT)),
            "prompt_a_sha256": file_sha256(PROMPT_A_PATH),
            "prompt_a_canonical_sha256": canonical_text_sha256(prompts[0]),
            "prompt_b_path": str(PROMPT_B_PATH.relative_to(REPOSITORY_ROOT)),
            "prompt_b_sha256": file_sha256(PROMPT_B_PATH),
            "prompt_b_canonical_sha256": canonical_text_sha256(prompts[1]),
            "experiment_009_runner_path": str(
                EXPERIMENT_009_RUNNER.relative_to(REPOSITORY_ROOT)
            ),
            "experiment_009_runner_sha256": file_sha256(
                EXPERIMENT_009_RUNNER
            ),
        },
        "original_b_reproduction_runs": [],
        "strengthened_endpoint_runs": [],
    }
    write_results(results_path, results)

    try:
        print("[original B reproduction]", flush=True)
        original_runs, original_token_count = run_endpoint_set(
            model=model,
            canvas=canvas,
            pair=original_pair,
            weights=(1.0,),
            seeds=args.seeds,
            output_dir=original_dir,
            canvas_length=args.canvas_length,
            steps=args.steps,
        )
        if original_token_count != ORIGINAL_PROMPT_TOKEN_COUNT:
            raise AssertionError(
                "original prompt token count changed: "
                f"{original_token_count} != {ORIGINAL_PROMPT_TOKEN_COUNT}"
            )
        for run in original_runs:
            reference_path = (
                REFERENCE_OUTPUT_DIR
                / f"dual-weight-1p0-seed-{run['seed']}-denoised.txt"
            )
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            generated_path = original_dir / run["denoised_output_file"]
            generated = generated_path.read_text(encoding="utf-8")
            reference = reference_path.read_text(encoding="utf-8")
            run["reference_output_file"] = str(
                reference_path.relative_to(REPOSITORY_ROOT)
            )
            run["reference_canonical_sha256"] = canonical_text_sha256(
                reference
            )
            run["canonical_reproduction_match"] = (
                canonical_text(generated) == canonical_text(reference)
            )
        results["original_b_reproduction_runs"] = original_runs
        results["original_prompt_token_count"] = original_token_count
        results["reproduction_controls_passed"] = all(
            run["canonical_reproduction_match"] for run in original_runs
        )
        write_results(results_path, results)
        if (
            not results["reproduction_controls_passed"]
            and args.model_id != LOCAL_TEST_MODEL_ID
        ):
            results["status"] = "blocked_reproduction"
            results["completed_at"] = datetime.now(UTC).isoformat()
            write_results(results_path, results)
            print(
                "Original outputs did not reproduce; strengthened runs skipped.",
                flush=True,
            )
            return

        print("[strengthened aligned endpoints]", flush=True)
        strengthened_runs, strengthened_token_count = run_endpoint_set(
            model=model,
            canvas=canvas,
            pair=explicit_pair,
            weights=(0.0, 1.0),
            seeds=args.seeds,
            output_dir=strengthened_dir,
            canvas_length=args.canvas_length,
            steps=args.steps,
        )
        if strengthened_token_count != STRENGTHENED_PROMPT_TOKEN_COUNT:
            raise AssertionError(
                "strengthened prompt token count changed: "
                f"{strengthened_token_count} != "
                f"{STRENGTHENED_PROMPT_TOKEN_COUNT}"
            )
        results["strengthened_endpoint_runs"] = strengthened_runs
        results["strengthened_prompt_token_count"] = strengthened_token_count
        results["strengthened_prompts_require_padding"] = False
        results["strengthened_A_endpoint_passed"] = endpoint_passes(
            strengthened_runs,
            weight=0.0,
            indicator="retains_A",
            expected_seed_count=len(args.seeds),
        )
        results["strengthened_B_endpoint_passed"] = endpoint_passes(
            strengthened_runs,
            weight=1.0,
            indicator="retains_B",
            expected_seed_count=len(args.seeds),
        )
    except Exception:
        results["status"] = "failed"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        raise

    results["status"] = (
        "completed_fixture"
        if args.model_id == LOCAL_TEST_MODEL_ID
        else "completed"
    )
    results["completed_at"] = datetime.now(UTC).isoformat()
    results["completed_run_count"] = (
        len(results["original_b_reproduction_runs"])
        + len(results["strengthened_endpoint_runs"])
    )
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
