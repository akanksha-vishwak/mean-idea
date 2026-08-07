from __future__ import annotations

import os

from mean_idea.diffusion_gemma import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    DiffusionGemma,
    DiffusionGemmaSettings,
)
from mean_idea.service import LatentModelService, create_app

MODEL_ID = os.environ.get("MEAN_IDEA_MODEL_ID", DEFAULT_MODEL_ID)

model = DiffusionGemma(
    DiffusionGemmaSettings(
        model_id=MODEL_ID,
        prompt=os.environ.get("MEAN_IDEA_PROMPT", DEFAULT_PROMPT),
        max_denoising_steps=int(os.environ.get("MEAN_IDEA_STEPS", "48")),
        dtype=os.environ.get("MEAN_IDEA_DTYPE", "auto"),
        device_map=os.environ.get("MEAN_IDEA_DEVICE_MAP", "auto"),
    )
)
app = create_app(LatentModelService(model, MODEL_ID))