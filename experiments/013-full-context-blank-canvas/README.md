# Experiment 013: Full-context blank-canvas prompting
- Question: Does supplying all frozen pair context in one prompt improve blank-canvas idea composition?
- Inputs/control: Experiment 012's three pairs, seeds 42-44, 8 steps, 96 tokens, and its frozen concise direct-prompt baseline.
- Changes: Add this README, `spec.json`, `prompt-template.txt`, and `run.py`; production model code is unchanged.
- Run: `uv run python experiments/013-full-context-blank-canvas/run.py`.
- Outputs: Preserve exact prompts, raw generations, lexical gates, hashes, timings, model revision, and the baseline comparison.
- Interpretation: Compare full-context hybrids with 8/9 concise-prompt hybrids; commit/push implementation and reviewed results separately.
