# Experiment 012: Non-Sorting Composition
- Question: Do the sorting failures of hard and per-step-logit midpoints replicate on endpoint-valid non-sorting pairs?
- Inputs: The three Experiment 011 pairs selected solely by their frozen 6/6 endpoint gates.
- Controls: Verify the committed Experiment 011 manifest, model revision, prompt hashes, passing IDs, and token counts before generation.
- Changes: Add this README, `spec.json`, `direct-prompts.json`, and `run.py`; production model code remains unchanged.
- Run: `uv run python experiments/012-nonsorting-composition/run.py`.
- Outputs: Save 27 midpoint/direct ideas, projections, traces, checksums, and pair-level hybrid counts.
