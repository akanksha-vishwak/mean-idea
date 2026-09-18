# Research communication package

This directory is rebuilt directly from the committed experiment manifests and raw text outputs. It contains no cached evidence snapshot.

From the repository root, run:

```powershell
uv run --with python-pptx --with matplotlib python deliverables\build.py
```

The build fails if mandatory evidence for Experiments 001–012 is absent or malformed. The exact raw JSON for the earlier September 16 partial-scaffold and scaffold-handoff pilot is preserved under `evidence\prior-pilot`; it is source evidence, not a generated snapshot or cache.

Generated outputs:

- `presentation.pptx` — 22-slide presentation.
- `poster.html` — printable landscape poster.
- `technical-report.html` — detailed methods, results, raw-output excerpts, and provenance.
- `answers-to-sergiy.md` — direct answers to the proposal questions.
- `assets\*.png` — plots generated from raw JSON fields.
- `evidence\prior-pilot\` — raw JSON needed to reproduce the earlier partial-scaffold and handoff claims.
