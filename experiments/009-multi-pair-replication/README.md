# Experiment 009: Multi-Pair Replication
- Question: Does the numerical-mixing failure versus direct-prompt success replicate beyond one sorting pair?
- Inputs: Two preselected sorting pairs and one frozen non-sorting byte-histogram pair, with aligned source prompts.
- Controls: Three seeds per endpoint; only pairs passing all six controls proceed to midpoint and direct-prompt runs.
- Changes: Add this README, `spec.json`, `pairs.json`, and `run.py`; production model code remains unchanged.
- Run: `uv run python experiments/009-multi-pair-replication/run.py`.
- Outputs: Save every raw idea, projection, trace, failure, checksum, and pair-level summary; commit results only after review.
