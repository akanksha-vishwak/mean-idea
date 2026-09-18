# Experiment 004: Controlled Prompt-State Weight Sweep
Question: Does any arithmetic mixture of bounded-counting and Numba prompt states preserve both?
Changes: add two explicit prompts, `spec.json`, and `run.py`; no core library code changes.
Inputs: identical empty canvases; weights `0.0–1.0` by `0.1`; seeds `42,43,44`.
Run: `uv run python experiments/004-controlled-prompt-weight-sweep/run.py`
Outputs: up to 33 raw `.txt` generations plus metadata in `results/results.json`.
Expect: endpoints must pass first; a hybrid must retain counting sort and Numba/JIT.
