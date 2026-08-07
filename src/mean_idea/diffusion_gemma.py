from __future__ import annotations

from dataclasses import dataclass

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
    max_input_tokens: int = 256
    quantization_chunk_size: int = 8192
    dtype: str = "auto"
    device_map: str = "auto"


class DiffusionGemma:
    """DiffusionGemma adapter for contextual canvas interpolation."""

    def __init__(self, settings: DiffusionGemmaSettings | None = None) -> None:
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        self.settings = settings or DiffusionGemmaSettings()
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.settings.model_id,
            dtype=self.settings.dtype,
            device_map=self.settings.device_map,
        )
        self.model.eval()
        self.canvas_length = int(self.model.config.canvas_length)
        self.prompt_inputs = self._base_prompt_inputs()
        self.max_input_tokens = _validate_max_input_tokens(
            self.settings.max_input_tokens,
            canvas_length=self.canvas_length,
            context_length=int(self.model.config.text_config.max_position_embeddings),
            prompt_length=self.prompt_inputs["input_ids"].shape[-1],
        )

    @torch.inference_mode()
    def text_to_embedding(self, text: str) -> TextEmbedding:
        """Return denoiser hidden states for sequential fixed-length canvases."""

        encoded = self.processor.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_input_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        token_ids = encoded.input_ids.to(self.model.device)
        token_mask = encoded.attention_mask.to(self.model.device)
        chunks = []
        for start in range(0, self.max_input_tokens, self.canvas_length):
            inputs = self._prompt_inputs(
                token_ids[:, :start],
                token_mask[:, :start],
            )
            outputs = self.model.model(
                **inputs,
                decoder_input_ids=token_ids[:, start : start + self.canvas_length],
            )
            chunks.append(outputs.last_hidden_state[0].detach().float().cpu())

        values = torch.cat(chunks, dim=0)
        return TextEmbedding(values.detach().float().cpu())

    @torch.inference_mode()
    def embedding_to_text(self, embedding: TextEmbedding) -> str:
        """Quantize and refine sequential embedding canvases."""

        self._validate_shape(embedding)
        generated_chunks = []
        for start in range(0, self.max_input_tokens, self.canvas_length):
            chunk = embedding.values[start : start + self.canvas_length]
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
                max_new_tokens=self.canvas_length,
                max_denoising_steps=self.settings.max_denoising_steps,
            )
            sequences = output.sequences if hasattr(output, "sequences") else output
            generated = sequences[
                :, prompt_length : prompt_length + self.canvas_length
            ]
            generated_chunks.append(generated)
            if self._contains_eos(generated):
                break

        all_generated = torch.cat(generated_chunks, dim=-1)[0]
        return self.processor.decode(
            all_generated, skip_special_tokens=True
        ).strip()

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

    def _validate_shape(self, embedding: TextEmbedding) -> None:
        expected = (
            self.max_input_tokens,
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


def _validate_max_input_tokens(
    max_input_tokens: int,
    *,
    canvas_length: int,
    context_length: int,
    prompt_length: int,
) -> int:
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if max_input_tokens % canvas_length:
        raise ValueError(
            f"max_input_tokens must be a multiple of the {canvas_length}-token canvas"
        )
    if prompt_length + max_input_tokens > context_length:
        maximum = ((context_length - prompt_length) // canvas_length) * canvas_length
        raise ValueError(
            f"max_input_tokens exceeds the model context; maximum is {maximum}"
        )
    return max_input_tokens
