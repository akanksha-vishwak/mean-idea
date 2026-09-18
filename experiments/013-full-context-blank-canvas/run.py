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
from types import ModuleType
from typing import Any

import torch

from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "013-full-context-blank-canvas"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
PAIRS_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "011-nonsorting-endpoint-screen"
    / "pairs.json"
)
BASELINE_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "012-nonsorting-composition"
    / "results"
    / "results.json"
)
EXPERIMENT_012_RUNNER = (
    REPOSITORY_ROOT
    / "experiments"
    / "012-nonsorting-composition"
    / "run.py"
)
PROMPT_TEMPLATE_PATH = EXPERIMENT_DIR / "prompt-template.txt"
EXPECTED_PAIR_IDS = (
    "matrix-blocked-blas",
    "graph-csr-deque-bfs",
    "image-batched-vectorization",
)
DEFAULT_SEEDS = (42, 43, 44)
BASELINE_DIRECT_HYBRIDS = 8
BASELINE_DIRECT_RUNS = 9
MODEL_STARTUP_PROMPT = (
    "Generate one concise optimization idea from the supplied prompt context. "
    "Return only the optimization idea."
)


def load_experiment_012() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "experiment_012_runner",
        EXPERIMENT_012_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {EXPERIMENT_012_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E012 = load_experiment_012()


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
            "Run Experiment 013: full-context direct prompting with no "
            "parent-derived canvas state."
        )
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


def load_inputs() -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    pair_document = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_RESULTS_PATH.read_text(encoding="utf-8"))
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()

    pairs_by_id = {pair["id"]: pair for pair in pair_document["pairs"]}
    if not all(pair_id in pairs_by_id for pair_id in EXPECTED_PAIR_IDS):
        raise ValueError("one or more required pairs are missing")
    pairs = [pairs_by_id[pair_id] for pair_id in EXPECTED_PAIR_IDS]

    baseline_ids = tuple(
        pair_result["pair_id"] for pair_result in baseline["pair_results"]
    )
    if baseline_ids != EXPECTED_PAIR_IDS:
        raise ValueError("Experiment 012 pair order or selection changed")
    direct_runs = [
        run
        for pair_result in baseline["pair_results"]
        for run in pair_result["runs"]
        if run["method"] == "direct_diffusion_prompt"
    ]
    direct_hybrids = sum(
        run["lexical_indicators"]["candidate_hybrid"] for run in direct_runs
    )
    if (
        len(direct_runs) != BASELINE_DIRECT_RUNS
        or direct_hybrids != BASELINE_DIRECT_HYBRIDS
    ):
        raise ValueError("Experiment 012 direct-prompt baseline changed")
    if template.count("{parent_a}") != 1 or template.count("{parent_b}") != 1:
        raise ValueError("prompt template must contain each parent placeholder once")
    return pairs, template, baseline


def build_full_context_prompt(template: str, pair: dict[str, Any]) -> str:
    return template.format(
        parent_a=pair["source_prompt_a"],
        parent_b=pair["source_prompt_b"],
    )


def run_full_context(
    *,
    model: DiffusionGemma,
    pair: dict[str, Any],
    prompt: str,
    output_dir: Path,
    seed: int,
    canvas_length: int,
    steps: int,
) -> dict[str, Any]:
    output_path = output_dir / f"full-context-seed-{seed}.txt"
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    text = E012.generate_direct(
        model=model,
        prompt=prompt,
        seed=seed,
        canvas_length=canvas_length,
        steps=steps,
    )
    duration = time.monotonic() - started
    output_path.write_text(text + "\n", encoding="utf-8")
    return {
        "method": "full_context_blank_canvas_prompt",
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "output_file": output_path.name,
        "output_sha256": file_sha256(output_path),
        "lexical_indicators": E012.E009.lexical_indicators(text, pair),
    }


def baseline_pair_summary(
    baseline: dict[str, Any],
    pair_id: str,
) -> dict[str, Any]:
    pair_result = next(
        item for item in baseline["pair_results"] if item["pair_id"] == pair_id
    )
    direct_runs = [
        run
        for run in pair_result["runs"]
        if run["method"] == "direct_diffusion_prompt"
    ]
    return {
        "run_count": len(direct_runs),
        "hybrid_count": sum(
            run["lexical_indicators"]["candidate_hybrid"]
            for run in direct_runs
        ),
        "output_files": [
            {
                "seed": run["seed"],
                "path": (
                    f"experiments/012-nonsorting-composition/results/"
                    f"{pair_id}/{run['output_file']}"
                ),
                "sha256": run["output_sha256"],
            }
            for run in direct_runs
        ],
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    results_path = output_dir / "results.json"
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, template, baseline = load_inputs()
    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Pairs: {len(pairs)}; seeds: {args.seeds}", flush=True)
    print(
        f"Loading model once for {len(pairs) * len(args.seeds)} generations...",
        flush=True,
    )
    model = DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model_id,
            prompt=MODEL_STARTUP_PROMPT,
            max_denoising_steps=args.steps,
            canvas_length=args.canvas_length,
            multi_canvas=1,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    )
    baseline_model_revision = baseline["model"]["revision"]
    current_model_revision = getattr(model.model.config, "_commit_hash", None)
    if (
        args.model_id != LOCAL_TEST_MODEL_ID
        and current_model_revision != baseline_model_revision
    ):
        raise ValueError("model revision differs from Experiment 012")

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
            "method": "full_context_blank_canvas_prompt",
            "parent_derived_canvas_state": False,
            "new_generation_count": len(pairs) * len(args.seeds),
        },
        "inputs": {
            "pairs_path": str(PAIRS_PATH.relative_to(REPOSITORY_ROOT)),
            "pairs_sha256": file_sha256(PAIRS_PATH),
            "prompt_template_path": str(
                PROMPT_TEMPLATE_PATH.relative_to(REPOSITORY_ROOT)
            ),
            "prompt_template_sha256": file_sha256(PROMPT_TEMPLATE_PATH),
            "baseline_results_path": str(
                BASELINE_RESULTS_PATH.relative_to(REPOSITORY_ROOT)
            ),
            "baseline_results_sha256": file_sha256(BASELINE_RESULTS_PATH),
        },
        "baseline": {
            "experiment_id": baseline["experiment_id"],
            "model_revision": baseline_model_revision,
            "hybrid_count": BASELINE_DIRECT_HYBRIDS,
            "run_count": BASELINE_DIRECT_RUNS,
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
            prompt = build_full_context_prompt(template, pair)
            prompt_path = pair_dir / "full-context-prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            prompt_token_count = int(
                model._base_prompt_inputs(prompt)["input_ids"].shape[-1]
            )
            pair_result: dict[str, Any] = {
                "pair_id": pair_id,
                "workload": pair["workload"],
                "expected_combined_concept": pair[
                    "expected_combined_concept"
                ],
                "prompt_file": prompt_path.name,
                "prompt_sha256": file_sha256(prompt_path),
                "prompt_character_count": len(prompt),
                "prompt_token_count": prompt_token_count,
                "baseline": baseline_pair_summary(baseline, pair_id),
                "runs": [],
            }
            results["pair_results"].append(pair_result)
            write_results(results_path, results)

            for seed in args.seeds:
                print(f"  [full context] seed={seed}", flush=True)
                pair_result["runs"].append(
                    run_full_context(
                        model=model,
                        pair=pair,
                        prompt=prompt,
                        output_dir=pair_dir,
                        seed=seed,
                        canvas_length=args.canvas_length,
                        steps=args.steps,
                    )
                )
                write_results(results_path, results)

            hybrid_count = sum(
                run["lexical_indicators"]["candidate_hybrid"]
                for run in pair_result["runs"]
            )
            run_count = len(pair_result["runs"])
            hybrid_rate = hybrid_count / run_count
            baseline_rate = (
                pair_result["baseline"]["hybrid_count"]
                / pair_result["baseline"]["run_count"]
            )
            pair_result["summary"] = {
                "hybrid_count": hybrid_count,
                "run_count": run_count,
                "hybrid_rate": hybrid_rate,
                "concise_baseline_hybrid_rate": baseline_rate,
                "difference_from_concise_baseline_rate": (
                    hybrid_rate - baseline_rate
                ),
            }
            write_results(results_path, results)
    except Exception:
        results["status"] = "failed"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        raise

    full_context_hybrids = sum(
        pair["summary"]["hybrid_count"] for pair in results["pair_results"]
    )
    completed_run_count = sum(
        len(pair["runs"]) for pair in results["pair_results"]
    )
    results["summary"] = {
        "pair_count": len(results["pair_results"]),
        "completed_run_count": completed_run_count,
        "full_context_hybrid_count": full_context_hybrids,
        "full_context_hybrid_rate": (
            full_context_hybrids / completed_run_count
        ),
        "concise_baseline_hybrid_count": BASELINE_DIRECT_HYBRIDS,
        "concise_baseline_run_count": BASELINE_DIRECT_RUNS,
        "concise_baseline_hybrid_rate": (
            BASELINE_DIRECT_HYBRIDS / BASELINE_DIRECT_RUNS
        ),
        "difference_from_concise_baseline_rate": (
            full_context_hybrids / completed_run_count
            - BASELINE_DIRECT_HYBRIDS / BASELINE_DIRECT_RUNS
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
