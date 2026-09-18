# Assessment

- Experiment 011 completed all 24 frozen endpoint generations with verified raw-file checksums. Three of four non-sorting pairs passed both three-seed endpoint gates.
- `matrix-blocked-blas` passed 6/6: A preserved NumPy `matmul` and BLAS, while B preserved blocked or tiled multiplication, cache locality, and accumulation. B also introduced Numba, but the predefined source technique remained intact.
- `regex-mmap-scan` was blocked: its compiled-regex A endpoint passed only 1/3 because two outputs omitted `re.compile` and drifted toward mmap-based scanning. Its mmap B endpoint passed 3/3.
- `graph-csr-deque-bfs` passed 6/6: A preserved `deque`, `popleft`, and BFS queue semantics; B preserved CSR, `indices`, `indptr`, and contiguous neighbor traversal. B also introduced Numba without losing the required representation.
- `image-batched-vectorization` passed 6/6: A preserved NumPy broadcasting and float32 vectorization; B preserved fixed batches, temporary-memory release, and output order. B added multiprocessing but retained the frozen batching technique.
- Initial projections were fragmented for every pair, yet denoising recovered both endpoints reliably for the three passing pairs. Endpoint decodability therefore extends beyond sorting when prompts and concepts are sufficiently controllable.
- The passing pairs were selected solely by the frozen 6/6 endpoint rule: `matrix-blocked-blas`, `graph-csr-deque-bfs`, and `image-batched-vectorization`. The blocked regex pair must remain excluded from midpoint interpretation.
- The next admissible experiment is one frozen midpoint comparison over all three passing pairs, using the exact Experiment 011 prompts and settings, with hard projection, per-step linear-logit mixing, and direct textual combination.
- Scope: DiffusionGemma revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, repository revision `198c2aa`, four public synthetic non-sorting workloads, seeds 42-44, eight denoising steps, and 96-token canvases.
