# Assessment

- The parent states are positively related rather than opposed: global cosine similarity is 0.542, mean position-wise cosine is 0.580, and none of the 64 positions has negative cosine similarity.
- The arithmetic midpoint retains 87.8% of the parents' mean global norm, so destructive vector cancellation does not explain the failure.
- At zero denoising steps, weight 0.0 contains fixed-range counting-sort fragments and weight 1.0 contains Numba `@njit` and interpreted-overhead fragments.
- The zero-step midpoint already contains neither technique; it decodes as damaged generic sorting text before diffusion refinement begins.
- The transition is visible around the midpoint: weight 0.4 retains damaged counting-sort language, weight 0.5 retains neither parent, weight 0.6 introduces a damaged `@njit` fragment, and weight 0.7 clearly enters the Numba basin.
- Experiment 004's eight-step outputs sharpen the same regions into counting-sort, unrelated merge-sort, and Numba ideas respectively; denoising does not recover a hybrid.
- Therefore the first observed semantic loss occurs during or before independent per-position vocabulary projection, not gradually during later denoising. The current evidence cannot distinguish whether the continuous midpoint lacks compositional information or the argmax projection discards distributed information.
- Scope: one validated prompt pair, one model revision, final prompt-conditioned canvas states, arithmetic interpolation, and independent highest-scoring-token projection.
