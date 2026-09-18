# Experiment 005: Projection and Geometry Diagnostic
- Question: Is the off-source midpoint already created by vocabulary projection, or only by later denoising?
- Inputs: Experiment 004's validated empty canvas and explicit counting-sort and Numba prompts at weights 0.0-1.0.
- Controls: Reuse and hash the three eight-step outputs per weight from Experiment 004; zero-step projection itself is deterministic.
- Changes: Add this README, `spec.json`, and `run.py`; do not alter model or interpolation code.
- Run: `uv run python experiments/005-projection-geometry-diagnostic/run.py`.
- Outputs: Save raw projected text, token IDs, cosine/norm geometry, token-overlap measures, and control references; commit/push code now, results later.
