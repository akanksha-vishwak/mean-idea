# Experiment 008: Dual-Prompt Logit Mixing
- Question: Can combining A- and B-conditioned logits at every denoising step compose counting sort with Numba?
- Inputs: Equal-token aligned A/B prompts and hard-projected states; logit weights 0.0, 0.25, 0.5, 0.75, and 1.0.
- Controls: Three seeds per endpoint, plus three exact aligned hard-projection and three direct diffusion-prompt midpoint baselines.
- Changes: Add this README, two prompt files, `spec.json`, and `run.py`; no production model or interpolation code changes.
- Run: `uv run python experiments/008-dual-prompt-logit-mixing/run.py`.
- Outputs: Save initial canvases, denoised ideas, per-step mixing traces, controls, failures, and provenance; commit/push code now and results after review.
