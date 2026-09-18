# Experiment 010: Histogram Endpoint Specificity
- Question: Can explicit ownership wording reliably preserve the frozen multiprocessing chunk-and-sum endpoint?
- Inputs: The Experiment 009 B prompt as a reproduction control and strengthened equal-token A/B histogram prompts.
- Controls: Seeds 42, 43, and 44; original outputs must canonically match Experiment 009 before interpreting strengthened endpoints.
- Changes: Add this README, `spec.json`, two prompt files, and `run.py`; production model code remains unchanged.
- Run: `uv run python experiments/010-histogram-endpoint-specificity/run.py`.
- Outputs: Save nine raw endpoint ideas, traces, checksums, reproduction matches, and endpoint gates; no midpoint is tested.
