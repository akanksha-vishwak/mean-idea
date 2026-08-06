from __future__ import annotations

from dataclasses import dataclass

import torch

from mean_idea.api import TextEmbedding

DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
DEFAULT_PROMPT = (
    "Refine the supplied token canvas into one coherent Python program. "
    "Return only the program, without Markdown fences or explanation."
)


@dataclass(frozen=True, slots=True)
class DiffusionGemmaSettings:
    model_id: str = DEFAULT_MODEL_ID
    prompt: str = DEFAULT_PROMPT
    max_denoising_steps: int = 48
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

    @torch.inference_mode()
    def text_to_embedding(self, text: str) -> TextEmbedding:
        """Return the denoiser hidden states for one fixed-length text canvas."""

        token_ids = self.processor.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.canvas_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        outputs = self.model.model(
            **self._prompt_inputs(),
            decoder_input_ids=token_ids.to(self.model.device),
        )
        values = outputs.last_hidden_state[0]
        return TextEmbedding(values.detach().float().cpu())

    @torch.inference_mode()
    def embedding_to_text(self, embedding: TextEmbedding) -> str:
        """Quantize an embedding canvas, then refine it with diffusion."""

        self._validate_shape(embedding)
        decoder_input_ids = self._project_to_tokens(embedding.values).unsqueeze(0)

        inputs = self._prompt_inputs()
        prompt_length = inputs["input_ids"].shape[-1]
        output = self.model.generate(
            **inputs,
            decoder_input_ids=decoder_input_ids.to(self.model.device),
            max_new_tokens=self.canvas_length,
            max_denoising_steps=self.settings.max_denoising_steps,
        )
        sequences = output.sequences if hasattr(output, "sequences") else output
        generated = sequences[0, prompt_length : prompt_length + self.canvas_length]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _prompt_inputs(self):
        return self.processor.apply_chat_template(
            [{"role": "user", "content": self.settings.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    def _validate_shape(self, embedding: TextEmbedding) -> None:
        expected = (
            self.canvas_length,
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
