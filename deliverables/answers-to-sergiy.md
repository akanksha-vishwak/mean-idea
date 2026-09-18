# Answers to Sergiy’s proposal

## Attribution key

The first five sections below are items **explicitly requested or proposed by Sergiy/the submission**. The later section titled **“Our proposed future work”** contains research directions developed after the original proposal and must not be attributed to Sergiy.

## 1. Prompt/internal representation mixing from blank canvases — explicit proposal item

**Status: tested negative for the implemented final-state mechanisms.** The experiments began with blank output canvases and mixed prompt-conditioned numerical states or output logits. DiffusionGemma decoded validated endpoints and generated fluent text, but fluency was not faithful composition. Experiment 004 produced **0/27** hybrids at intermediate weights. Across endpoint-valid Experiments 008, 009, and 012, final-state projection and linear output-logit mixing produced **0/42** strict hybrids.

This result is bounded to final-state interpolation/projection and linear output-logit mixing. It does not test every possible internal representation intervention.

## 2. Darwin ideas and programs — explicit proposal item

**Status: partly executed.** Darwin-derived optimization ideas supplied the initial sorting parents, and Experiment 001 attempted code-state controls. However, all Experiment 001 generated programs were invalid Python, and no numerical mixture established a useful program or a Darwin benchmark improvement. The later nonsorting pairs were public synthetic optimization tasks used to test whether the mechanism generalized beyond sorting.

## 3. Partial denoising followed by larger autoregressive completion — explicit proposal item

**Status: tested in the earlier pilot; it added no useful information.** The partial-denoising study tested **15** midpoint outputs and found **0** useful partial scaffolds. In the explicit handoff, scaffold-only completions recovered neither parent in **0/4** cases. Parents plus scaffold succeeded **4/4**, but the same parents without the scaffold had already succeeded **3/3**. The larger autoregressive model succeeded because it received the parent ideas, not because the partial diffusion output preserved a useful combination.

## 4. Bayesian optimization in representation space — explicit proposal item

**Status: not justified for the tested coordinate.** Experiment 004’s endpoint-controlled sweep moved from an A-like basin through a broad unrelated region to a B-like basin, with no hybrid region. The aggregate numerical result was **0/42** hybrids. Bayesian optimization would therefore search a discontinuous transition rather than a demonstrated smooth, useful idea space.

## 5. Eventual Darwin search-stage integration and code optimization — explicit proposal item

**Status: not justified for the failed numerical mechanism.** It would add complexity and GPU cost without a demonstrated synthesis benefit. A simpler direct text-based idea synthesizer remains viable: direct diffusion prompting produced **17/18** strict hybrids across endpoint-valid sorting, matrix, graph, and image pairs. Program implementation, correctness testing, benchmarking, and fitness improvement inside Darwin remain untested.

## Follow-up suggested by Sergiy on September 18: everything in the prompt, blank canvas

**Status: tested successfully, with no aggregate improvement over the concise prompt.** Experiment 013 placed both complete frozen parent records in one prompt and supplied no parent embedding, hidden state, token canvas, or logits. Full-context prompting produced **8/9** strict hybrids, exactly tying Experiment 012's concise blank-canvas baseline at **8/9**.

Matrix multiplication remained 3/3. Graph BFS improved from 2/3 to 3/3, while image normalization declined from 3/3 to 2/3 because one output omitted the frozen output-order requirement. Human inspection found the core two-technique combination in that ninth output, but the preregistered primary result remains 8/9. These public synthetic tasks contain complete parent descriptions rather than complete parent programs, so a full-program-context study remains technically distinct.

## What was not completed—and why

These were **implementation aspirations from the proposal**, contingent on the research mechanism first passing scientific gates. They were not forgotten:

- **No production Darwin sampling-stage integration.** The tested numerical synthesis mechanism had no demonstrated benefit, so integration was not scientifically justified.
- **No Bayesian optimizer run.** The required smooth-coordinate gate failed: interpolation crossed a broad unrelated region rather than a useful continuous idea space.
- **No end-to-end generated-program compilation, test, and benchmark campaign.** The numerical mechanism produced no useful synthesized idea to advance, and the initial code-state outputs were invalid.

The underlying research questions were still addressed: the tested representation mixing was negative; partial scaffold/handoff added no useful information; Bayesian optimization was not justified; and the failed mechanism did not merit Darwin integration. Direct text-based idea synthesis remains a viable separate path.

## What succeeded?

- Reproducible full-model execution with recorded prompts, hashes, seeds, revisions, runtime, and raw outputs.
- Parent endpoint decoding when prompts were specific enough.
- A diagnostic explanation stronger than “the vectors cancelled”: global cosine was **0.542**, no positions had negative cosine, and midpoint norm retention was **87.8%**.
- Evidence that parent alternatives often remain below top-1, identifying independent vocabulary projection as a likely bottleneck.
- Direct diffusion prompting composed both parents in **17/18** strict runs across four workloads.
- Experiment 011 found **3/4** endpoint-valid nonsorting pairs: matrix multiplication, graph BFS, and image normalization; regex was blocked.
- Experiment 012 replicated the numerical failure on all three endpoint-valid nonsorting pairs. Matrix direct prompting passed 3/3, graph 2/3 strict, and image 3/3.
- Experiment 013 tested Sergiy's full-context blank-canvas follow-up and tied the concise baseline at 8/9 strict hybrids. Graph improved to 3/3 while image declined to 2/3.
- The remaining graph direct output was human-identifiable as deque plus CSR, but a damaged `indptr` token correctly made it a strict automated-gate failure.

## Our proposed future work — not Sergiy’s explicit asks

These directions were developed later by the project team:

- Internal transformer-layer injection and layer selection.
- Prompt KV-cache mixing.
- Learned sequence-aware or alignment-aware mixing.
- A preregistered confirmatory study with held-out pairs, multiple workloads, blinded judges, uncertainty estimates, and program benchmarks.

They are technically distinct future studies, not unfinished versions of Sergiy’s original final-state mixing experiment.

## Scope statement

These are **exploratory/pilot** findings for the exact DiffusionGemma revision, prompts, final-canvas states, linear output-logit mixing, projection methods, weights, seeds, and workloads tested. They do **not** establish that all diffusion language-model composition methods fail. The negative result is bounded to final-state interpolation/projection and linear output-logit mixing.

## Provenance

- Numbered experiment evidence: `experiments/001-*` through `experiments/013-*`.
- Earlier partial-scaffold and handoff evidence: `deliverables/evidence/prior-pilot/partial-scaffolds-20260916/results.json`, `deliverables/evidence/prior-pilot/partial-scaffolds-20260916/assessment.json`, `deliverables/evidence/prior-pilot/scaffold-handoff-20260916/results.json`, and `deliverables/evidence/prior-pilot/scaffold-handoff-20260916/assessment.json`.
