from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MethodType
from typing import Iterator

import torch

from mean_idea.api import TextEmbedding

# Random 4.27M-parameter fixture for local plumbing tests; not useful for quality.
LOCAL_TEST_MODEL_ID = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"
# BF16 reference model: about 51 GB of weights and requires a larger accelerator.
REFERENCE_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
# A useful sub-32-GB option is unsloth/diffusiongemma-26B-A4B-it-GGUF Q4_K_M
# (16.8 GB), but it requires llama.cpp's diffusion runner, not this adapter.
# No useful 2-4 GB DiffusionGemma checkpoint is currently published.
DEFAULT_MODEL_ID = LOCAL_TEST_MODEL_ID
DEFAULT_PROMPT = (
    "Refine the supplied token canvas into one coherent Python program. "
    "Return only the program, without Markdown fences or explanation."
)


@dataclass(frozen=True, slots=True)
class DiffusionGemmaSettings:
    model_id: str = DEFAULT_MODEL_ID
    prompt: str = DEFAULT_PROMPT
    max_denoising_steps: int = 48
    canvas_length: int = 256
    multi_canvas: int = 1
    quantization_chunk_size: int = 8192
    dtype: str = "auto"
    device_map: str = "auto"


class DiffusionGemma:
    """DiffusionGemma adapter for contextual canvas interpolation."""

    def __init__(self, settings: DiffusionGemmaSettings | None = None) -> None:
        from transformers import (
            AutoProcessor,
            DiffusionGemmaConfig,
            DiffusionGemmaForBlockDiffusion,
        )

        self.settings = settings or DiffusionGemmaSettings()
        _validate_canvas_settings(
            canvas_length=self.settings.canvas_length,
            multi_canvas=self.settings.multi_canvas,
        )
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)
        config = DiffusionGemmaConfig.from_pretrained(self.settings.model_id)
        config.canvas_length = self.settings.canvas_length
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.settings.model_id,
            config=config,
            dtype=self.settings.dtype,
            device_map=self.settings.device_map,
        )
        self.model._denoising_step = MethodType(
            _memory_efficient_denoising_step,
            self.model,
        )
        self.model.eval()
        self._canvas_lock = RLock()
        self.prompt_inputs = self._base_prompt_inputs()
        _input_length(
            canvas_length=self.settings.canvas_length,
            multi_canvas=self.settings.multi_canvas,
            context_length=int(self.model.config.text_config.max_position_embeddings),
            prompt_length=self.prompt_inputs["input_ids"].shape[-1],
        )

    @torch.inference_mode()
    def text_to_embedding(
        self,
        text: str,
        *,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> TextEmbedding:
        """Return denoiser hidden states for sequential fixed-length canvases."""

        with self._using_canvas_settings(
            canvas_length,
            multi_canvas,
        ) as (active_canvas_length, input_length):
            encoded = self.processor.tokenizer(
                text,
                add_special_tokens=True,
                max_length=input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            token_ids = encoded.input_ids.to(self.model.device)
            token_mask = encoded.attention_mask.to(self.model.device)
            chunks = []
            for start in range(0, input_length, active_canvas_length):
                inputs = self._prompt_inputs(
                    token_ids[:, :start],
                    token_mask[:, :start],
                )
                outputs = self.model.model(
                    **inputs,
                    decoder_input_ids=token_ids[
                        :, start : start + active_canvas_length
                    ],
                )
                chunks.append(outputs.last_hidden_state[0].detach().float().cpu())

            values = torch.cat(chunks, dim=0)
        return TextEmbedding(values.detach().float().cpu())

    @torch.inference_mode()
    def embedding_to_text(
        self,
        embedding: TextEmbedding,
        *,
        canvas_length: int | None = None,
        multi_canvas: int | None = None,
    ) -> str:
        """Quantize and refine sequential embedding canvases."""

        with self._using_canvas_settings(
            canvas_length,
            multi_canvas,
        ) as (active_canvas_length, input_length):
            self._validate_shape(embedding, input_length)
            generated_chunks = []
            for start in range(0, input_length, active_canvas_length):
                chunk = embedding.values[start : start + active_canvas_length]
                decoder_input_ids = self._project_to_tokens(chunk).unsqueeze(0)
                generated_context = (
                    torch.cat(generated_chunks, dim=-1)
                    if generated_chunks
                    else torch.empty(
                        (1, 0), dtype=torch.long, device=self.model.device
                    )
                )
                inputs = self._prompt_inputs(
                    generated_context,
                    torch.ones_like(generated_context),
                )
                prompt_length = inputs["input_ids"].shape[-1]
                output = self.model.generate(
                    **inputs,
                    decoder_input_ids=decoder_input_ids.to(self.model.device),
                    max_new_tokens=active_canvas_length,
                    max_denoising_steps=self.settings.max_denoising_steps,
                )
                sequences = (
                    output.sequences if hasattr(output, "sequences") else output
                )
                generated = sequences[
                    :, prompt_length : prompt_length + active_canvas_length
                ]
                generated_chunks.append(generated)
                if self._contains_eos(generated):
                    break

            all_generated = torch.cat(generated_chunks, dim=-1)[0]
        return self.processor.decode(
            all_generated, skip_special_tokens=True
        ).strip()

    @contextmanager
    def _using_canvas_settings(
        self,
        canvas_length: int | None,
        multi_canvas: int | None,
    ) -> Iterator[tuple[int, int]]:
        active_canvas_length = (
            self.settings.canvas_length
            if canvas_length is None
            else canvas_length
        )
        active_multi_canvas = (
            self.settings.multi_canvas
            if multi_canvas is None
            else multi_canvas
        )
        input_length = _input_length(
            canvas_length=active_canvas_length,
            multi_canvas=active_multi_canvas,
            context_length=int(self.model.config.text_config.max_position_embeddings),
            prompt_length=self.prompt_inputs["input_ids"].shape[-1],
        )
        with self._canvas_lock:
            previous_canvas_length = self.model.config.canvas_length
            self.model.config.canvas_length = active_canvas_length
            try:
                yield active_canvas_length, input_length
            finally:
                self.model.config.canvas_length = previous_canvas_length

    def _base_prompt_inputs(self):
        return self.processor.apply_chat_template(
            [{"role": "user", "content": self.settings.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    def _prompt_inputs(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.cat(
                [self.prompt_inputs["input_ids"], context_ids], dim=-1
            ),
            "attention_mask": torch.cat(
                [self.prompt_inputs["attention_mask"], context_mask], dim=-1
            ),
        }

    def _contains_eos(self, token_ids: torch.Tensor) -> bool:
        eos_token_ids = self.model.generation_config.eos_token_id
        if eos_token_ids is None:
            return False
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        return any((token_ids == token_id).any().item() for token_id in eos_token_ids)

    def _validate_shape(
        self,
        embedding: TextEmbedding,
        input_length: int,
    ) -> None:
        expected = (
            input_length,
            int(self.model.config.text_config.hidden_size),
        )
        if tuple(embedding.values.shape) != expected:
            raise ValueError(
                f"embedding has shape {tuple(embedding.values.shape)}; expected {expected}"
            )

    def _project_to_tokens(self, values: torch.Tensor) -> torch.Tensor:
        return _highest_score_token_ids(
            values,
            self.model.lm_head.weight,
            chunk_size=self.settings.quantization_chunk_size,
        )


def _highest_score_token_ids(
    values: torch.Tensor,
    vocabulary: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Apply the tied LM head without materializing all vocabulary logits."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if values.ndim != 2 or vocabulary.ndim != 2:
        raise ValueError("values and vocabulary must both be matrices")
    if values.shape[1] != vocabulary.shape[1]:
        raise ValueError("embedding hidden sizes do not match")

    device = vocabulary.device
    queries = values.to(device=device, dtype=torch.float32)
    best_score = torch.full(
        (queries.shape[0],), -torch.inf, device=device, dtype=torch.float32
    )
    best_ids = torch.zeros(queries.shape[0], device=device, dtype=torch.long)

    for start in range(0, vocabulary.shape[0], chunk_size):
        candidates = vocabulary[start : start + chunk_size].float()
        chunk_score, chunk_ids = (queries @ candidates.T).max(dim=1)
        improved = chunk_score > best_score
        best_score = torch.where(improved, chunk_score, best_score)
        best_ids = torch.where(improved, chunk_ids + start, best_ids)

    return best_ids


def _sample_tokens_and_entropy(
    logits: torch.Tensor,
    *,
    sequence_chunk_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_chunk_size <= 0:
        raise ValueError("sequence_chunk_size must be positive")

    sampled_chunks = []
    entropy_chunks = []
    vocab_size = logits.shape[-1]
    for start in range(0, logits.shape[1], sequence_chunk_size):
        chunk = logits[:, start : start + sequence_chunk_size]
        probabilities = torch.softmax(chunk, dim=-1, dtype=torch.float32)
        sampled = torch.multinomial(
            probabilities.reshape(-1, vocab_size),
            num_samples=1,
        ).reshape(chunk.shape[0], chunk.shape[1])
        log_normalizer = torch.logsumexp(chunk.float(), dim=-1)
        expected_logit = (probabilities * chunk.float()).sum(dim=-1)
        sampled_chunks.append(sampled)
        entropy_chunks.append(log_normalizer - expected_logit)

    return (
        torch.cat(sampled_chunks, dim=1),
        torch.cat(entropy_chunks, dim=1),
    )


def _memory_efficient_denoising_step(
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
    cur_step = torch.tensor(
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
    processed_logits = logits_processor(
        input_ids,
        decoder_outputs.logits,
        cur_step=cur_step,
    )
    denoiser_canvas, token_entropy = _sample_tokens_and_entropy(
        processed_logits
    )
    new_argmax_canvas = torch.argmax(processed_logits, dim=-1)

    sorted_entropy, sorted_indices = torch.sort(token_entropy, dim=-1)
    cumulative_entropy = torch.cumsum(sorted_entropy, dim=-1)
    sorted_selection = (
        cumulative_entropy - sorted_entropy <= sampler.entropy_bound
    )
    sampler.accepted_token_mask = torch.scatter(
        torch.zeros_like(sorted_selection),
        dim=-1,
        index=sorted_indices,
        src=sorted_selection,
    )
    accepted_canvas = torch.where(
        sampler.accepted_token_mask,
        denoiser_canvas,
        current_canvas,
    )
    new_current_canvas = sampler.renoise_canvas(
        accepted_canvas,
        cur_step,
    ).clone()

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
            processed_logits[finished_denoising] = self_conditioning_logits[
                finished_denoising
            ]

        if diffusion_stopping_criteria.stability_threshold == 0:
            stable = torch.ones_like(finished_denoising)
        else:
            history = diffusion_stopping_criteria.argmax_canvas_history
            if history is None:
                history = torch.full(
                    (
                        diffusion_stopping_criteria.stability_threshold,
                        *new_argmax_canvas.shape,
                    ),
                    -1,
                    dtype=new_argmax_canvas.dtype,
                    device=new_argmax_canvas.device,
                )
            stable = (history == new_argmax_canvas[None]).all(dim=-1).all(dim=0)
            history = torch.roll(history, shifts=-1, dims=0)
            history[-1] = new_argmax_canvas
            diffusion_stopping_criteria.argmax_canvas_history = history

        confident = (
            token_entropy.mean(dim=-1)
            < diffusion_stopping_criteria.confidence_threshold
        )
        finished_denoising |= stable & confident

    embeddings_dtype = model.model.decoder.embed_tokens.weight.dtype
    self_conditioning_logits = processed_logits.to(embeddings_dtype)
    return (
        new_current_canvas,
        new_argmax_canvas,
        self_conditioning_logits,
        finished_denoising,
    )


def _validate_canvas_settings(
    *,
    canvas_length: int,
    multi_canvas: int,
) -> None:
    if not isinstance(canvas_length, int) or isinstance(canvas_length, bool):
        raise ValueError("canvas_length must be a positive integer")
    if canvas_length <= 0:
        raise ValueError("canvas_length must be a positive integer")
    if not isinstance(multi_canvas, int) or isinstance(multi_canvas, bool):
        raise ValueError("multi_canvas must be a positive integer")
    if multi_canvas <= 0:
        raise ValueError("multi_canvas must be a positive integer")


def _input_length(
    *,
    canvas_length: int,
    multi_canvas: int,
    context_length: int,
    prompt_length: int,
) -> int:
    _validate_canvas_settings(
        canvas_length=canvas_length,
        multi_canvas=multi_canvas,
    )
    input_length = canvas_length * multi_canvas
    if prompt_length + input_length > context_length:
        maximum = (context_length - prompt_length) // canvas_length
        raise ValueError(
            f"multi_canvas exceeds the model context; maximum is {maximum}"
        )
    return input_length
