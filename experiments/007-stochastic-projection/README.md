# Experiment 007: Stochastic Projection
- Question: Can sampling retained top-20 alternatives avoid the incoherent hard-top-1 midpoint and produce a hybrid?
- Inputs: Experiment 006 candidate scores for weights 0.0, 0.5, and 1.0; temperatures 0.5, 1.0, and 2.0.
- Controls: Three seeds per endpoint and temperature; only temperatures preserving both endpoints proceed to five midpoint seeds.
- Changes: Add this README, `spec.json`, and `run.py`; do not alter model, interpolation, or denoising code.
- Run: `uv run python experiments/007-stochastic-projection/run.py`.
- Outputs: Save every sampled initial canvas, denoised idea, rank metadata, failure, and gate decision; commit/push code now and results after review.
