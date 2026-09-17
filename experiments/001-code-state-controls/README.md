# Experiment 001: Code-State Controls
Question: Can the original pipeline preserve each source program before A and B are mixed?
Changes: `run.py` runs A+A, B+B, and A+B; `spec.json` defines the controls and success rules.
Inputs: pure-Python counting sort A, NumPy `bincount`/`repeat` sort B, and the shared sorting prompt.
Run: `uv run python experiments/001-code-state-controls/run.py`
Outputs: three readable `.txt` generations plus metadata in `results/results.json`.
Expect: A+A retains counting sort, B+B retains NumPy details, and A+B is judged only if both controls work.
