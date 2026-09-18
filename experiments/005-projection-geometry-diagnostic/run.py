from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from mean_idea.api import TextEmbedding
from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "005-projection-geometry-diagnostic"
CONTROL_EXPERIMENT_ID = "004-controlled-prompt-weight-sweep"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PROMPT_A_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / CONTROL_EXPERIMENT_ID
    / "prompt-a-counting.txt"
)
PROMPT_B_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / CONTROL_EXPERIMENT_ID
    / "prompt-b-numba.txt"
)
DEFAULT_CONTROL_RESULTS_DIR = (
    REPOSITORY_ROOT / "experiments" / CONTROL_EXPERIMENT_ID / "results"
)
GENERATION_PROMPT = (
    "Generate one concise optimization idea for the Python function my_sort. "
    "Preserve the specific algorithmic and implementation details represented "
    "in the supplied canvas. Do not use list.sort or sorted. Return only the "
    "optimization idea."
)
WEIGHTS = tuple(round(index / 10, 1) for index in range(11))


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 005: measure parent-state geometry and inspect the "
            "deterministic zero-step vocabulary projection at every weight."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--control-results-dir",
        type=Path,
        default=DEFAULT_CONTROL_RESULTS_DIR,
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--skip-control-comparison",
        action="store_true",
        help="skip Experiment 004 controls; intended only for tiny-fixture validation",
    )
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
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


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        functional.cosine_similarity(
            left.reshape(1, -1).float(),
            right.reshape(1, -1).float(),
        ).item()
    )


def state_geometry(
    values: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weight: float,
) -> dict[str, float]:
    values_float = values.float()
    left_float = left.float()
    right_float = right.float()
    left_norm = float(torch.linalg.vector_norm(left_float).item())
    right_norm = float(torch.linalg.vector_norm(right_float).item())
    mixed_norm = float(torch.linalg.vector_norm(values_float).item())
    linear_norm_reference = (1.0 - weight) * left_norm + weight * right_norm
    return {
        "global_norm": mixed_norm,
        "norm_retention_vs_linear_endpoint_norms": (
            mixed_norm / linear_norm_reference
            if linear_norm_reference
            else 0.0
        ),
        "global_cosine_to_A": cosine(values_float, left_float),
        "global_cosine_to_B": cosine(values_float, right_float),
        "distance_to_A": float(
            torch.linalg.vector_norm(values_float - left_float).item()
        ),
        "distance_to_B": float(
            torch.linalg.vector_norm(values_float - right_float).item()
        ),
    }


def parent_geometry(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_float = left.float()
    right_float = right.float()
    position_cosines = functional.cosine_similarity(
        left_float,
        right_float,
        dim=1,
    )
    midpoint = 0.5 * (left_float + right_float)
    mean_parent_norm = 0.5 * (
        torch.linalg.vector_norm(left_float)
        + torch.linalg.vector_norm(right_float)
    )
    return {
        "global_cosine": cosine(left_float, right_float),
        "position_cosine": {
            "mean": float(position_cosines.mean().item()),
            "minimum": float(position_cosines.min().item()),
            "maximum": float(position_cosines.max().item()),
            "negative_position_count": int((position_cosines < 0).sum().item()),
        },
        "global_norm_A": float(torch.linalg.vector_norm(left_float).item()),
        "global_norm_B": float(torch.linalg.vector_norm(right_float).item()),
        "midpoint_global_norm": float(
            torch.linalg.vector_norm(midpoint).item()
        ),
        "midpoint_norm_retention_vs_mean_parent_norm": float(
            (torch.linalg.vector_norm(midpoint) / mean_parent_norm).item()
        ),
    }


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
    return {
        "retains_A": counting or (frequency and bounded),
        "retains_B": numba and jit,
        "retains_both": (counting or (frequency and bounded)) and numba and jit,
    }


def load_control_references(
    *,
    control_results_dir: Path,
    model_id: str,
    canvas_length: int,
) -> list[dict[str, Any]]:
    manifest_path = control_results_dir / "results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != CONTROL_EXPERIMENT_ID:
        raise ValueError(f"{manifest_path} is not an Experiment 004 manifest")
    if manifest.get("status") != "completed":
        raise ValueError(f"{manifest_path} is not completed")
    if manifest.get("model_id") != model_id:
        raise ValueError(
            "control model does not match this run: "
            f"{manifest.get('model_id')!r} != {model_id!r}"
        )
    settings = manifest.get("settings", {})
    if settings.get("canvas_length") != canvas_length:
        raise ValueError("control canvas length does not match this run")
    if settings.get("max_denoising_steps") != 8:
        raise ValueError("control denoising steps do not match Experiment 005")
    if settings.get("generation_prompt") != GENERATION_PROMPT:
        raise ValueError("control generation prompt does not match this run")
    if settings.get("weights") != list(WEIGHTS):
        raise ValueError("control weight grid does not match this run")
    if settings.get("seeds") != [42, 43, 44]:
        raise ValueError("control seeds do not match Experiment 005")
    if manifest.get("endpoint_controls_passed") is not True:
        raise ValueError("Experiment 004 endpoint controls did not pass")

    expected_prompt_hashes = {
        "A": canonical_text_sha256(PROMPT_A_PATH),
        "B": canonical_text_sha256(PROMPT_B_PATH),
    }
    for name, expected_hash in expected_prompt_hashes.items():
        if manifest["prompts"][name]["sha256"] != expected_hash:
            raise ValueError(f"control prompt {name} does not match this run")

    references = []
    for run in manifest["runs"]:
        if run.get("status") != "completed":
            raise ValueError("control manifest contains a failed run")
        output_path = control_results_dir / run["output_file"]
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        references.append(
            {
                "weight": run["weight"],
                "seed": run["seed"],
                "steps": settings["max_denoising_steps"],
                "output_path": str(output_path.relative_to(REPOSITORY_ROOT)),
                "output_sha256": file_sha256(output_path),
                "lexical_indicators": run["lexical_indicators"],
            }
        )
    expected_conditions = {
        (weight, seed)
        for weight in WEIGHTS
        for seed in (42, 43, 44)
    }
    observed_conditions = {
        (reference["weight"], reference["seed"])
        for reference in references
    }
    if observed_conditions != expected_conditions:
        raise ValueError("control manifest does not contain the expected 33 runs")
    return references


def token_overlap(
    token_ids: torch.Tensor,
    endpoint_a: torch.Tensor,
    endpoint_b: torch.Tensor,
) -> dict[str, int | float]:
    if token_ids.shape != endpoint_a.shape or token_ids.shape != endpoint_b.shape:
        raise ValueError("projected token sequences must have identical shapes")
    same_a = token_ids == endpoint_a
    same_b = token_ids == endpoint_b
    same_neither = ~(same_a | same_b)
    length = token_ids.numel()
    return {
        "position_count": length,
        "same_as_A_count": int(same_a.sum().item()),
        "same_as_B_count": int(same_b.sum().item()),
        "same_as_neither_count": int(same_neither.sum().item()),
        "same_as_A_fraction": float(same_a.float().mean().item()),
        "same_as_B_fraction": float(same_b.float().mean().item()),
        "same_as_neither_fraction": float(same_neither.float().mean().item()),
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
    print("Loading the model once for deterministic projection...", flush=True)

    model = DiffusionGemma(
        DiffusionGemmaSettings(
            model_id=args.model_id,
            prompt=GENERATION_PROMPT,
            max_denoising_steps=0,
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

    control_references: list[dict[str, Any]] = []
    if not args.skip_control_comparison:
        control_references = load_control_references(
            control_results_dir=args.control_results_dir.resolve(),
            model_id=args.model_id,
            canvas_length=args.canvas_length,
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
            "projection_steps": 0,
            "weights": list(WEIGHTS),
            "mixing": "global arithmetic final-canvas-state interpolation",
            "projection": "independent highest-scoring LM-head token per position",
            "generation_prompt": GENERATION_PROMPT,
            "control_comparison_skipped": args.skip_control_comparison,
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
                "canonical_text_sha256": canonical_text_sha256(path),
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
        "parent_geometry": parent_geometry(
            embeddings["A"].values,
            embeddings["B"].values,
        ),
        "control_references": control_references,
        "projection_runs": [],
    }
    write_results(results_path, results)

    projected: dict[float, torch.Tensor] = {}
    mixed_embeddings: dict[float, TextEmbedding] = {}
    try:
        for index, weight in enumerate(WEIGHTS, start=1):
            print(
                f"[projection {index}/{len(WEIGHTS)}] weight={weight:.1f}",
                flush=True,
            )
            mixed = interpolate_embeddings(
                embeddings["A"],
                embeddings["B"],
                weight,
            )
            token_ids = model._project_to_tokens(mixed.values).cpu()
            text = model.processor.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
            weight_label = f"{weight:.1f}".replace(".", "p")
            output_path = output_dir / f"weight-{weight_label}-projection.txt"
            output_path.write_text(text + "\n", encoding="utf-8")
            mixed_embeddings[weight] = mixed
            projected[weight] = token_ids

        endpoint_a = projected[0.0]
        endpoint_b = projected[1.0]
        for weight in WEIGHTS:
            output_path = output_dir / (
                f"weight-{weight:.1f}".replace(".", "p") + "-projection.txt"
            )
            text = output_path.read_text(encoding="utf-8").rstrip("\n")
            results["projection_runs"].append(
                {
                    "weight": weight,
                    "status": "completed",
                    "output_file": output_path.name,
                    "output_sha256": file_sha256(output_path),
                    "projected_token_ids": projected[weight].tolist(),
                    "token_overlap_with_endpoints": token_overlap(
                        projected[weight],
                        endpoint_a,
                        endpoint_b,
                    ),
                    "state_geometry": state_geometry(
                        mixed_embeddings[weight].values,
                        embeddings["A"].values,
                        embeddings["B"].values,
                        weight,
                    ),
                    "lexical_indicators": lexical_indicators(text),
                }
            )
            write_results(results_path, results)
    except Exception as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        write_results(results_path, results)
        raise

    endpoint_agreement = endpoint_a == endpoint_b
    results["endpoint_projected_token_agreement"] = {
        "position_count": endpoint_a.numel(),
        "same_token_count": int(endpoint_agreement.sum().item()),
        "same_token_fraction": float(endpoint_agreement.float().mean().item()),
    }
    results["status"] = (
        "completed_fixture"
        if args.skip_control_comparison or args.model_id == LOCAL_TEST_MODEL_ID
        else "completed"
    )
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
