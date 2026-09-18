# Experiment 006: Top-Candidate Projection
- Question: At the failed midpoint, do both endpoint tokens remain below top-1, or has the projection distribution already lost them?
- Inputs: The Experiment 004 blank canvas and prompts at weights 0.0, 0.4, 0.5, 0.6, and 1.0.
- Controls: Every top-1 token must reproduce Experiment 005 exactly; endpoint-distinct positions are analyzed separately from shared boilerplate.
- Changes: Add this README, `spec.json`, and `run.py`; do not alter model, interpolation, or denoising code.
- Run: `uv run python experiments/006-top-candidate-projection/run.py`.
- Outputs: Save top-20 tokens, scores, endpoint ranks, and summaries without denoising; commit/push code now and results after review.
