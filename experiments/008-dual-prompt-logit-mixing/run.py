from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from mean_idea.api import TextEmbedding
from mean_idea.diffusion_gemma import (
    LOCAL_TEST_MODEL_ID,
    REFERENCE_MODEL_ID,
    DiffusionGemma,
    DiffusionGemmaSettings,
    _memory_efficient_denoising_step,
    _sample_tokens_and_entropy,
)


EXPERIMENT_ID = "008-dual-prompt-logit-mixing"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
BLANK_CANVAS_PATH = REPOSITORY_ROOT / "tests" / "example" / "blank.txt"
PROMPT_A_PATH = EXPERIMENT_DIR / "prompt-a-aligned.txt"
PROMPT_B_PATH = EXPERIMENT_DIR / "prompt-b-aligned.txt"
GENERATION_PROMPT = (
    "Generate one concise optimization idea for the Python function my_sort. "
    "Preserve the specific algorithmic and implementation details represented "
    "in the supplied canvas. Do not use list.sort or sorted. Return only the "
    "optimization idea."
)
DIRECT_COMBINATION_PROMPT = (
    "Combine these two optimization techniques into one coherent optimization "
    "idea for my_sort: (A) bounded counting sort using a fixed frequency array "
    "for integers 0-9999; (B) compile the custom sorting loops with Numba "
    "@njit to remove Python interpreter overhead. The result must explicitly "
    "use both techniques. Do not use list.sort or sorted."
)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
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
            "Run Experiment 008: combine source-conditioned logits at every "
            "DiffusionGemma denoising step."
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
        help="continue after endpoint failure; intended only for fixture testing",
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    counting = "counting sort" in lowered or "counting-sort" in lowered
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


def prompt_batch(model: DiffusionGemma, prompts: tuple[str, str]):
    encoded = [model._base_prompt_inputs(prompt) for prompt in prompts]
    lengths = [int(item["input_ids"].shape[-1]) for item in encoded]
    if lengths[0] != lengths[1]:
        raise ValueError(
            f"dual prompts must have equal token lengths, got {lengths}"
        )
    input_ids = torch.cat([item["input_ids"] for item in encoded], dim=0)
    attention_mask = torch.cat(
        [item["attention_mask"] for item in encoded],
        dim=0,
    )
    if not bool(attention_mask.all().item()):
        raise ValueError("dual prompts must not require padding")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _dual_prompt_denoising_step(
    model,
    *,
    decoder_forward,
    current_canvas,
    argmax_canvas,
    input_ids,
    decoder_position_ids,
    self_conditioning_logits,
    mask_mapping,
    past_key_values,
    finished_denoising,
    cur_step,
    sampler,
    logits_processor,
    diffusion_stopping_criteria,
    **model_kwargs,
):
    if current_canvas.shape[0] != 2:
        return _memory_efficient_denoising_step(
            model,
            decoder_forward=decoder_forward,
            current_canvas=current_canvas,
            argmax_canvas=argmax_canvas,
            input_ids=input_ids,
            decoder_position_ids=decoder_position_ids,
            self_conditioning_logits=self_conditioning_logits,
            mask_mapping=mask_mapping,
            past_key_values=past_key_values,
            finished_denoising=finished_denoising,
            cur_step=cur_step,
            sampler=sampler,
            logits_processor=logits_processor,
            diffusion_stopping_criteria=diffusion_stopping_criteria,
            **model_kwargs,
        )

    weight = float(model._dual_prompt_mix_weight)
    cur_step_tensor = torch.tensor(
        cur_step,
        device=current_canvas.device,
        dtype=torch.int32,
    )
    torch.compiler.cudagraph_mark_step_begin()
    decoder_outputs = decoder_forward(
        decoder_input_ids=current_canvas,
        self_conditioning_logits=self_conditioning_logits,
        decoder_attention_mask=mask_mapping,
        past_key_values=past_key_values,
        decoder_position_ids=decoder_position_ids,
        **model_kwargs,
    )
    processed = logits_processor(
        input_ids,
        decoder_outputs.logits,
        cur_step=cur_step_tensor,
    )
    source_a = processed[0:1].float()
    source_b = processed[1:2].float()
    combined = torch.lerp(source_a, source_b, weight)

    denoiser_canvas, token_entropy = _sample_tokens_and_entropy(combined)
    shared_argmax = torch.argmax(combined, dim=-1)
    sorted_entropy, sorted_indices = torch.sort(token_entropy, dim=-1)
    cumulative_entropy = torch.cumsum(sorted_entropy, dim=-1)
    sorted_selection = (
        cumulative_entropy - sorted_entropy <= sampler.entropy_bound
    )
    shared_accept_mask = torch.scatter(
        torch.zeros_like(sorted_selection),
        dim=-1,
        index=sorted_indices,
        src=sorted_selection,
    )
    shared_current = current_canvas[0:1]
    accepted = torch.where(
        shared_accept_mask,
        denoiser_canvas,
        shared_current,
    )
    random_canvas = sampler.initialize_canvas(
        batch_size=1,
        device=current_canvas.device,
    )
    shared_next = torch.where(
        ~shared_accept_mask,
        random_canvas,
        accepted,
    )

    new_current_canvas = shared_next.repeat(2, 1)
    new_argmax_canvas = shared_argmax.repeat(2, 1)
    sampler.accepted_token_mask = shared_accept_mask.repeat(2, 1)
    combined_batch = combined.to(
        model.model.decoder.embed_tokens.weight.dtype
    ).repeat(2, 1, 1)

    if diffusion_stopping_criteria is not None:
        if finished_denoising.any():
            new_argmax_canvas = torch.where(
                finished_denoising[:, None],
                argmax_canvas,
                new_argmax_canvas,
            )
            new_current_canvas = torch.where(
                finished_denoising[:, None],
                current_canvas,
                new_current_canvas,
            )
            combined_batch = torch.where(
                finished_denoising[:, None, None],
                self_conditioning_logits,
                combined_batch,
            )
        finished_denoising |= diffusion_stopping_criteria(
            new_argmax_canvas,
            combined_batch,
        )

    top_a = torch.argmax(source_a, dim=-1)
    top_b = torch.argmax(source_b, dim=-1)
    top_combined = shared_argmax
    model._dual_prompt_trace.append(
        {
            "step": int(cur_step),
            "source_top_1_agreement_fraction": float(
                (top_a == top_b).float().mean().item()
            ),
            "combined_top_1_equals_A_fraction": float(
                (top_combined == top_a).float().mean().item()
            ),
            "combined_top_1_equals_B_fraction": float(
                (top_combined == top_b).float().mean().item()
            ),
            "combined_top_1_equals_neither_fraction": float(
                ((top_combined != top_a) & (top_combined != top_b))
                .float()
                .mean()
                .item()
            ),
            "accepted_token_count": int(shared_accept_mask.sum().item()),
        }
    )
    return (
        new_current_canvas,
        new_argmax_canvas,
        combined_batch,
        finished_denoising,
    )


def generate_dual_prompt(
    *,
    model: DiffusionGemma,
    inputs,
    initial_token_ids: torch.Tensor,
    weight: float,
    seed: int,
    canvas_length: int,
    steps: int,
) -> tuple[str, list[dict[str, Any]]]:
    if initial_token_ids.ndim != 1 or initial_token_ids.numel() != canvas_length:
        raise ValueError("initial token IDs must match one canvas")
    set_seed(seed)
    model.model._dual_prompt_mix_weight = weight
    model.model._dual_prompt_trace = []
    output = model.model.generate(
        **inputs,
        decoder_input_ids=initial_token_ids.unsqueeze(0)
        .repeat(2, 1)
        .to(model.model.device),
        max_new_tokens=canvas_length,
        max_denoising_steps=steps,
        disable_compile=True,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    prompt_length = inputs["input_ids"].shape[-1]
    generated = sequences[:, prompt_length : prompt_length + canvas_length]
    if not torch.equal(generated[0], generated[1]):
        raise AssertionError("dual-prompt rows produced different canvases")
    text = model.processor.decode(
        generated[0],
        skip_special_tokens=True,
    ).strip()
    return text, list(model.model._dual_prompt_trace)


def generate_direct_prompt(
    *,
    model: DiffusionGemma,
    seed: int,
    canvas_length: int,
    steps: int,
) -> str:
    set_seed(seed)
    inputs = model._base_prompt_inputs(DIRECT_COMBINATION_PROMPT)
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
    weight_label = f"{weight:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    stem = f"weight-{weight_label}-seed-{seed}"
    initial_path = output_dir / f"{stem}-initial.txt"
    output_path = output_dir / f"{stem}-denoised.txt"
    trace_path = output_dir / f"{stem}-trace.json"
    initial_path.write_text(initial_text + "\n", encoding="utf-8")
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    text, trace = generate_dual_prompt(
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
        "initial_lexical_indicators": lexical_indicators(initial_text),
        "denoised_lexical_indicators": lexical_indicators(text),
    }


def run_direct_condition(
    *,
    model: DiffusionGemma,
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
        "lexical_indicators": lexical_indicators(text),
    }


def run_hard_projection_condition(
    *,
    model: DiffusionGemma,
    embedding: TextEmbedding,
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
    initial_path = output_dir / f"hard-projection-seed-{seed}-initial.txt"
    output_path = output_dir / f"hard-projection-seed-{seed}-denoised.txt"
    initial_path.write_text(initial_text + "\n", encoding="utf-8")
    set_seed(seed)
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
        "weight": 0.5,
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "initial_output_file": initial_path.name,
        "initial_output_sha256": file_sha256(initial_path),
        "denoised_output_file": output_path.name,
        "denoised_output_sha256": file_sha256(output_path),
        "initial_lexical_indicators": lexical_indicators(initial_text),
        "denoised_lexical_indicators": lexical_indicators(text),
    }


def controls_pass(runs: list[dict[str, Any]]) -> bool:
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

    canvas = BLANK_CANVAS_PATH.read_text(encoding="utf-8")
    if canvas:
        raise ValueError(f"{BLANK_CANVAS_PATH} must be empty")
    prompts = {
        "A": PROMPT_A_PATH.read_text(encoding="utf-8"),
        "B": PROMPT_B_PATH.read_text(encoding="utf-8"),
    }

    print(f"Experiment: {EXPERIMENT_ID}", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Logit weights: {WEIGHTS}", flush=True)
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
    model.model._denoising_step = MethodType(
        _dual_prompt_denoising_step,
        model.model,
    )
    dual_inputs = prompt_batch(model, (prompts["A"], prompts["B"]))
    prompt_token_count = int(dual_inputs["input_ids"].shape[-1])
    embeddings = {
        name: model.text_to_embedding(
            canvas,
            prompt=prompt,
            canvas_length=args.canvas_length,
            multi_canvas=1,
        )
        for name, prompt in prompts.items()
    }
    mixed_embeddings = {
        weight: interpolate_embeddings(
            embeddings["A"],
            embeddings["B"],
            weight,
        )
        for weight in WEIGHTS
    }

    runs: list[dict[str, Any]] = []
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
            "weights": list(WEIGHTS),
            "seeds": list(args.seeds),
            "initialization": "hard projection of arithmetic final-state interpolation",
            "per_step_mixing": "linear interpolation of A- and B-prompt processed logits",
            "generation_prompt": GENERATION_PROMPT,
            "direct_combination_prompt": DIRECT_COMBINATION_PROMPT,
            "dual_prompt_batch_shape": list(dual_inputs["input_ids"].shape),
            "aligned_prompt_token_count": prompt_token_count,
            "aligned_prompts_require_padding": False,
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
            }
            for name, path in (("A", PROMPT_A_PATH), ("B", PROMPT_B_PATH))
        },
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
    try:
        for index, (weight, seed) in enumerate(endpoint_schedule, start=1):
            print(
                f"[control {index}/{len(endpoint_schedule)}] "
                f"weight={weight:.2f}, seed={seed}",
                flush=True,
            )
            try:
                run = run_dual_condition(
                    model=model,
                    inputs=dual_inputs,
                    embedding=mixed_embeddings[weight],
                    output_dir=output_dir,
                    weight=weight,
                    seed=seed,
                    canvas_length=args.canvas_length,
                    steps=args.steps,
                )
            except Exception as error:
                runs.append(
                    {
                        "method": "dual_prompt_logit_mixing",
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

        endpoint_controls_passed = controls_pass(runs)
        results["endpoint_controls_passed"] = endpoint_controls_passed
        write_results(results_path, results)
        if not endpoint_controls_passed and not args.allow_failed_controls:
            results["status"] = "blocked_controls"
            results["completed_at"] = datetime.now(UTC).isoformat()
            write_results(results_path, results)
            print(
                "Endpoint controls failed; intermediate and baseline runs skipped.",
                flush=True,
            )
            return

        intermediate_schedule = [
            (weight, seed)
            for weight in WEIGHTS[1:-1]
            for seed in args.seeds
        ]
        for index, (weight, seed) in enumerate(
            intermediate_schedule,
            start=1,
        ):
            print(
                f"[mix {index}/{len(intermediate_schedule)}] "
                f"weight={weight:.2f}, seed={seed}",
                flush=True,
            )
            try:
                run = run_dual_condition(
                    model=model,
                    inputs=dual_inputs,
                    embedding=mixed_embeddings[weight],
                    output_dir=output_dir,
                    weight=weight,
                    seed=seed,
                    canvas_length=args.canvas_length,
                    steps=args.steps,
                )
            except Exception as error:
                runs.append(
                    {
                        "method": "dual_prompt_logit_mixing",
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

        for index, seed in enumerate(args.seeds, start=1):
            print(
                f"[hard baseline {index}/{len(args.seeds)}] seed={seed}",
                flush=True,
            )
            try:
                run = run_hard_projection_condition(
                    model=model,
                    embedding=mixed_embeddings[0.5],
                    output_dir=output_dir,
                    seed=seed,
                    canvas_length=args.canvas_length,
                    steps=args.steps,
                )
            except Exception as error:
                runs.append(
                    {
                        "method": "aligned_hard_projection",
                        "weight": 0.5,
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

        for index, seed in enumerate(args.seeds, start=1):
            print(
                f"[direct baseline {index}/{len(args.seeds)}] seed={seed}",
                flush=True,
            )
            try:
                run = run_direct_condition(
                    model=model,
                    output_dir=output_dir,
                    seed=seed,
                    canvas_length=args.canvas_length,
                    steps=args.steps,
                )
            except Exception as error:
                runs.append(
                    {
                        "method": "direct_diffusion_prompt",
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
                f"saved {run['output_file']}",
                flush=True,
            )
    except Exception:
        results["status"] = "failed"
        results["completed_at"] = datetime.now(UTC).isoformat()
        write_results(results_path, results)
        raise

    results["candidate_hybrid_count"] = sum(
        run["denoised_lexical_indicators"]["candidate_hybrid"]
        for run in runs
        if run["method"] == "dual_prompt_logit_mixing"
        and 0.0 < run["weight"] < 1.0
    )
    results["direct_baseline_hybrid_count"] = sum(
        run["lexical_indicators"]["candidate_hybrid"]
        for run in runs
        if run["method"] == "direct_diffusion_prompt"
    )
    results["hard_projection_hybrid_count"] = sum(
        run["denoised_lexical_indicators"]["candidate_hybrid"]
        for run in runs
        if run["method"] == "aligned_hard_projection"
    )
    results["status"] = (
        "completed_fixture"
        if args.allow_failed_controls or args.model_id == LOCAL_TEST_MODEL_ID
        else "completed"
    )
    results["completed_at"] = datetime.now(UTC).isoformat()
    write_results(results_path, results)
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
