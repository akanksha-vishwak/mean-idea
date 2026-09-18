# Experiment 011: Non-Sorting Endpoint Screen
- Question: Which jointly frozen public non-sorting pairs have two reliable endpoints under one common template?
- Inputs: Matrix, regex/mmap, sparse-graph BFS, and image-normalization pairs with exact within-pair token alignment.
- Controls: Three seeds per A and B endpoint; every output is preserved and no pair may be replaced after screening.
- Changes: Add this README, `spec.json`, `pairs.json`, and `run.py`; production model code remains unchanged.
- Run: `uv run python experiments/011-nonsorting-endpoint-screen/run.py`.
- Outputs: Save 24 raw endpoint ideas, projections, traces, checksums, and frozen pair-level pass/fail decisions.
