# Experiment 013 assessment

- All nine preregistered full-model generations completed with the frozen model revision, seeds 42-44, eight denoising steps, and 96-token blank output canvases. All 13 result files and every manifest-recorded prompt/output hash were verified after retrieval.
- Full-context blank-canvas prompting produced 8/9 strict hybrids, exactly matching Experiment 012's concise direct-prompt baseline of 8/9. The aggregate strict-rate difference was therefore 0.
- Matrix multiplication remained 3/3: every output combined cache-sized blocking with BLAS-backed `np.matmul` and accumulation.
- Graph BFS improved from 2/3 strict to 3/3: every output combined `collections.deque`/`popleft` with CSR `indices`/`indptr` and contiguous-neighbor traversal.
- Image normalization declined from 3/3 strict to 2/3. Seed 44 still combined fixed-size batching, memory capping, NumPy broadcasting, and `float32`, but omitted the frozen output-order requirement and therefore remains a strict failure.
- Human inspection found the core two-technique combination in all nine outputs, but the preregistered primary result remains 8/9 because the rubric required every frozen lexical group. The leading `thought` marker appeared consistently and did not affect classification.
- Conclusion: Sergiy's prompt-only blank-canvas proposal works reliably on these public synthetic tasks, but supplying the complete parent records did not outperform a concise explicit combination prompt. Richer context changed which task missed an exact requirement rather than increasing aggregate reliability.
- Scope: this is exploratory evidence for three previously selected pairs and does not include complete parent programs or full Darwin state. It strengthens direct prompting as a viable baseline; it does not support numerical representation mixing, Bayesian optimization over the failed coordinate, or production Darwin integration.
