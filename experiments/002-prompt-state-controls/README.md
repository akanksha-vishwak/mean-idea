# Experiment 002: Prompt-State Controls
Question: With both canvases empty, can different idea prompts survive alone and combine?
Changes: add two readable prompts, `spec.json`, and `run.py`; no core library code changes.
Inputs: bounded counting-sort Prompt A, NumPy `bincount`/`repeat` Prompt B, and one shared empty canvas.
Run: `uv run python experiments/002-prompt-state-controls/run.py`
Outputs: three raw `.txt` generations plus metadata in `results/results.json`.
Expect: A+A and B+B preserve their techniques; A+B is useful only if it preserves both.
