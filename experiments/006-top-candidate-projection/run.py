from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from mean_idea.api import TextEmbedding
from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
)


EXPERIMENT_ID = "006-top-candidate-projection"
REFERENCE_EXPERIMENT_ID = "005-projection-geometry-diagnostic"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PROMPT_A_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "004-controlled-prompt-weight-sweep"
    / "prompt-a-counting.txt"
)
PROMPT_B_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "004-controlled-prompt-weight-sweep"
    / "prompt-b-numba.txt"
)
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
WEIGHTS = (0.0, 0.4, 0.5, 0.6, 1.0)
DEFAULT_TOP_K = 20
SUMMARY_CUTOFFS = (1, 5, 20)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 006: inspect top vocabulary candidates and endpoint "
            "token ranks around the failed interpolation midpoint."
        )
    )
    parser.add_argument("--model-id", default=REFERENCE_MODEL_ID)
    parser.add_argument("--canvas-length", type=positive_integer, default=64)
    parser.add_argument("--top-k", type=positive_integer, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=DEFAULT_REFERENCE_RESULTS_PATH,
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--skip-reference-control",
        action="store_true",
        help="skip Experiment 005 comparison; intended only for tiny-fixture validation",
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
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


@torch.inference_mode()
def top_k_token_ids_and_scores(
    values: torch.Tensor,
    vocabulary: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if values.ndim != 2 or vocabulary.ndim != 2:
        raise ValueError("values and vocabulary must both be matrices")
    if values.shape[1] != vocabulary.shape[1]:
        raise ValueError("embedding hidden sizes do not match")
    if top_k > vocabulary.shape[0]:
        raise ValueError("top_k cannot exceed vocabulary size")

    device = vocabulary.device
    queries = values.to(device=device, dtype=torch.float32)
    best_scores = torch.empty(
        (queries.shape[0], 0),
        device=device,
        dtype=torch.float32,
    )
    best_ids = torch.empty(
        (queries.shape[0], 0),
        device=device,
        dtype=torch.long,
    )

    for start in range(0, vocabulary.shape[0], chunk_size):
        candidates = vocabulary[start : start + chunk_size].float()
        logits = queries @ candidates.T
        chunk_top_k = min(top_k, logits.shape[1])
        chunk_scores, chunk_ids = torch.topk(
            logits,
            k=chunk_top_k,
            dim=1,
        )
        chunk_ids += start
        combined_scores = torch.cat((best_scores, chunk_scores), dim=1)
        combined_ids = torch.cat((best_ids, chunk_ids), dim=1)
        retained = min(top_k, combined_scores.shape[1])
        best_scores, retained_indices = torch.topk(
            combined_scores,
            k=retained,
            dim=1,
        )
        best_ids = torch.gather(combined_ids, 1, retained_indices)

    return best_ids.cpu(), best_scores.cpu()


@torch.inference_mode()
def selected_token_scores(
    values: torch.Tensor,
    vocabulary: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    if values.shape[0] != token_ids.numel():
        raise ValueError("one selected token ID is required per canvas position")
    device = vocabulary.device
    queries = values.to(device=device, dtype=torch.float32)
    selected = vocabulary[token_ids.to(device)].float()
    return (queries * selected).sum(dim=1).cpu()


def load_reference_results(
    *,
    path: Path,
    model_id: str,
    canvas_length: int,
) -> dict[str, Any]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("experiment_id") != REFERENCE_EXPERIMENT_ID:
        raise ValueError(f"{path} is not an Experiment 005 manifest")
    if reference.get("status") != "completed":
        raise ValueError(f"{path} is not completed")
    if reference["model"]["id"] != model_id:
        raise ValueError("reference model ID does not match this run")
    settings = reference["settings"]
    if settings["canvas_length"] != canvas_length:
        raise ValueError("reference canvas length does not match this run")
    if settings["generation_prompt"] != GENERATION_PROMPT:
        raise ValueError("reference generation prompt does not match this run")
    expected_hashes = {
        "A": canonical_text_sha256(PROMPT_A_PATH),
        "B": canonical_text_sha256(PROMPT_B_PATH),
    }
    for name, expected_hash in expected_hashes.items():
        recorded_hash = reference["prompts"][name].get(
            "canonical_text_sha256",
            reference["prompts"][name]["sha256"],
        )
        if recorded_hash != expected_hash:
            raise ValueError(f"reference prompt {name} does not match this run")

    runs = {
        float(run["weight"]): run
        for run in reference["projection_runs"]
        if float(run["weight"]) in WEIGHTS
    }
    if set(runs) != set(WEIGHTS):
        raise ValueError("reference manifest lacks a required target weight")
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
        "repository_revision": reference["repository_revision"],
        "model_revision": reference["model"]["revision"],
        "runs": runs,
    }


def rank_of(candidate_ids: torch.Tensor, target_id: int) -> int | None:
    matches = (candidate_ids == target_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return None
    return int(matches[0, 0].item()) + 1


def decode_token(model: DiffusionGemma, token_id: int) -> dict[str, Any]:
    tokenizer = model.processor.tokenizer
    return {
        "id": token_id,
        "piece": tokenizer.convert_ids_to_tokens(token_id),
        "decoded": tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def cutoff_summary(
    ranks_a: list[int | None],
    ranks_b: list[int | None],
    cutoff: int,
) -> dict[str, int | float]:
    count = len(ranks_a)
    present_a = [
        rank is not None and rank <= cutoff
        for rank in ranks_a
    ]
    present_b = [
        rank is not None and rank <= cutoff
        for rank in ranks_b
    ]
    both = [left and right for left, right in zip(present_a, present_b)]
    neither = [
        not left and not right
        for left, right in zip(present_a, present_b)
    ]
    denominator = count if count else 1
    return {
        "position_count": count,
        "A_present_count": sum(present_a),
        "B_present_count": sum(present_b),
        "both_present_count": sum(both),
        "neither_present_count": sum(neither),
        "A_present_fraction": sum(present_a) / denominator,
        "B_present_fraction": sum(present_b) / denominator,
        "both_present_fraction": sum(both) / denominator,
        "neither_present_fraction": sum(neither) / denominator,
    }


def found_rank_summary(ranks: list[int | None]) -> dict[str, int | float | None]:
    found = [rank for rank in ranks if rank is not None]
    return {
        "found_count": len(found),
        "missing_count": len(ranks) - len(found),
        "minimum": min(found) if found else None,
        "median": float(statistics.median(found)) if found else None,
        "maximum": max(found) if found else None,
    }


def analyze_weight(
    *,
    model: DiffusionGemma,
    values: torch.Tensor,
    endpoint_a: torch.Tensor,
    endpoint_b: torch.Tensor,
    top_ids: torch.Tensor,
    top_scores: torch.Tensor,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    discriminative = endpoint_a != endpoint_b
    discriminative_positions = (
        discriminative.nonzero(as_tuple=False).flatten().tolist()
    )
    endpoint_a_scores = selected_token_scores(
        values,
        model.model.lm_head.weight,
        endpoint_a,
    )
    endpoint_b_scores = selected_token_scores(
        values,
        model.model.lm_head.weight,
        endpoint_b,
    )

    detailed_positions = []
    ranks_a: list[int | None] = []
    ranks_b: list[int | None] = []
    favor_a = 0
    favor_b = 0
    ties = 0
    for position in range(endpoint_a.numel()):
        position_ids = top_ids[position]
        rank_a = rank_of(position_ids, int(endpoint_a[position].item()))
        rank_b = rank_of(position_ids, int(endpoint_b[position].item()))
        score_a = float(endpoint_a_scores[position].item())
        score_b = float(endpoint_b_scores[position].item())
        endpoint_tokens_differ = bool(discriminative[position].item())
        if endpoint_tokens_differ:
            ranks_a.append(rank_a)
            ranks_b.append(rank_b)
            if score_a > score_b:
                favor_a += 1
            elif score_b > score_a:
                favor_b += 1
            else:
                ties += 1

        candidates = []
        for rank in range(top_k):
            token_id = int(position_ids[rank].item())
            candidates.append(
                {
                    "rank": rank + 1,
                    **decode_token(model, token_id),
                    "score": float(top_scores[position, rank].item()),
                }
            )
        detailed_positions.append(
            {
                "position": position,
                "endpoint_tokens_differ": endpoint_tokens_differ,
                "endpoint_A": {
                    **decode_token(model, int(endpoint_a[position].item())),
                    "rank": rank_a,
                    "score": score_a,
                    "gap_from_top_1": float(
                        top_scores[position, 0].item() - score_a
                    ),
                },
                "endpoint_B": {
                    **decode_token(model, int(endpoint_b[position].item())),
                    "rank": rank_b,
                    "score": score_b,
                    "gap_from_top_1": float(
                        top_scores[position, 0].item() - score_b
                    ),
                },
                "candidates": candidates,
            }
        )

    effective_cutoffs = sorted(
        {min(cutoff, top_k) for cutoff in SUMMARY_CUTOFFS}
    )
    summary = {
        "endpoint_distinct_position_count": len(discriminative_positions),
        "endpoint_score_preference": {
            "A_higher_count": favor_a,
            "B_higher_count": favor_b,
            "tie_count": ties,
        },
        "endpoint_A_rank_within_top_k": found_rank_summary(ranks_a),
        "endpoint_B_rank_within_top_k": found_rank_summary(ranks_b),
        "availability_by_cutoff": {
            str(cutoff): cutoff_summary(ranks_a, ranks_b, cutoff)
            for cutoff in effective_cutoffs
        },
    }
    details = {
        "top_k": top_k,
        "positions": detailed_positions,
    }
    return summary, details


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
    reference = None
    if not args.skip_reference_control:
        reference = load_reference_results(
            path=args.reference_results.resolve(),
            model_id=args.model_id,
            canvas_length=args.canvas_length,
        )

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Weights: {WEIGHTS}", flush=True)
    print(f"Top-k: {args.top_k}", flush=True)
    print("Loading the model once for vocabulary scoring...", flush=True)

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
    model_revision = getattr(model.model.config, "_commit_hash", None)
    if (
        reference is not None
        and reference["model_revision"] is not None
        and model_revision != reference["model_revision"]
    ):
        raise ValueError("loaded model revision does not match Experiment 005")

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
            "denoising_steps": 0,
            "weights": list(WEIGHTS),
            "top_k": args.top_k,
            "mixing": "global arithmetic final-canvas-state interpolation",
            "generation_prompt": GENERATION_PROMPT,
            "reference_control_skipped": args.skip_reference_control,
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
            prompt_ids["A"],
            prompt_ids["B"],
        ),
        "embedding_shapes": {
            name: list(embedding.values.shape)
            for name, embedding in embeddings.items()
        },
        "reference": (
            {
                key: value
                for key, value in reference.items()
                if key != "runs"
            }
            if reference is not None
            else None
        ),
        "runs": [],
    }
    write_json(results_path, results)

    mixed_embeddings = {
        weight: interpolate_embeddings(
            embeddings["A"],
            embeddings["B"],
            weight,
        )
        for weight in WEIGHTS
    }
    scored: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    try:
        for index, weight in enumerate(WEIGHTS, start=1):
            print(
                f"[score {index}/{len(WEIGHTS)}] weight={weight:.1f}",
                flush=True,
            )
            top_ids, top_scores = top_k_token_ids_and_scores(
                mixed_embeddings[weight].values,
                model.model.lm_head.weight,
                top_k=args.top_k,
                chunk_size=model.settings.quantization_chunk_size,
            )
            adapter_top_1 = model._project_to_tokens(
                mixed_embeddings[weight].values
            ).cpu()
            if not torch.equal(top_ids[:, 0], adapter_top_1):
                raise AssertionError(
                    f"top-k rank 1 does not reproduce adapter projection at {weight}"
                )
            if reference is not None:
                reference_top_1 = torch.tensor(
                    reference["runs"][weight]["projected_token_ids"],
                    dtype=torch.long,
                )
                if not torch.equal(top_ids[:, 0], reference_top_1):
                    raise AssertionError(
                        "top-k rank 1 does not reproduce Experiment 005 "
                        f"at weight {weight}"
                    )
            scored[weight] = (top_ids, top_scores)

        endpoint_a = scored[0.0][0][:, 0]
        endpoint_b = scored[1.0][0][:, 0]
        for weight in WEIGHTS:
            top_ids, top_scores = scored[weight]
            summary, details = analyze_weight(
                model=model,
                values=mixed_embeddings[weight].values,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                top_ids=top_ids,
                top_scores=top_scores,
                top_k=args.top_k,
            )
            weight_label = f"{weight:.1f}".replace(".", "p")
            top_1_path = output_dir / f"weight-{weight_label}-top1.txt"
            details_path = (
                output_dir / f"weight-{weight_label}-top-candidates.json"
            )
            top_1_text = model.processor.decode(
                top_ids[:, 0],
                skip_special_tokens=True,
            ).strip()
            top_1_path.write_text(top_1_text + "\n", encoding="utf-8")
            write_json(
                details_path,
                {
                    "weight": weight,
                    "top_1_output_file": top_1_path.name,
                    **details,
                },
            )
            results["runs"].append(
                {
                    "weight": weight,
                    "status": "completed",
                    "top_1_output_file": top_1_path.name,
                    "top_1_output_sha256": file_sha256(top_1_path),
                    "candidate_file": details_path.name,
                    "candidate_file_sha256": file_sha256(details_path),
                    "summary": summary,
                }
            )
            write_json(results_path, results)
    except Exception as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        write_json(results_path, results)
        raise

    results["endpoint_distinct_position_count"] = int(
        (endpoint_a != endpoint_b).sum().item()
    )
    results["status"] = (
        "completed_fixture"
        if args.skip_reference_control or args.model_id == LOCAL_TEST_MODEL_ID
        else "completed"
    )
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_json(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
