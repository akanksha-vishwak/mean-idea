# Experiment 003: Prompt-Specificity Diagnostic
Question: Where do the named Source B details `NumPy`, `np.bincount`, and `np.repeat` disappear?
Changes: add two B prompts, `spec.json`, and `run.py`; no core library code changes.
Inputs: identical empty canvases with baseline or explicit B prompts; denoising steps `0,1,2,4,8`.
Run: `uv run python experiments/003-prompt-specificity-diagnostic/run.py`
Outputs: eleven raw `.txt` generations plus metadata in `results/results.json`.
Expect: locate loss before/during denoising and test whether direct decoder prompt visibility restores B.
