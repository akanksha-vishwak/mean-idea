# Assessment

- The control gate passed: every newly computed top-1 token exactly reproduced both the adapter projection and Experiment 005 at all five tested weights.
- The A and B endpoint projections differ at 40 of 64 positions; the remaining 24 shared positions are excluded from the primary availability statistics.
- At weight 0.5, the B endpoint token scores higher at 25 of 40 distinct positions and A scores higher at 15, confirming a moderate B-side bias rather than disappearance of A.
- At the midpoint top-1, A wins 8 positions, B wins 22, and a third token wins 10; this position-wise patchwork produces the damaged zero-step sequence.
- Within the top five, A is available at 25 positions, B at 32, and both at 21. Within the top 20, A is available at 38 positions, B at all 40, and both at 38.
- Technique-bearing fragments also survive below top-1: counting-sort pieces are commonly rank 2, while Numba pieces such as `umba`, `` `@ ``, and `nj` appear around ranks 3-7 at the relevant midpoint positions.
- Therefore Experiment 005's continuous midpoint has not simply lost both parents before projection. The independent hard top-1 choice discards competing parent alternatives and assembles an incoherent cross-position token sequence.
- This does not prove that independent top-k sampling will produce a coherent hybrid: endpoint fragments have unequal score gaps and multi-token techniques require coordinated choices across positions. The justified next test is a controlled soft or stochastic projection with endpoint gates, followed by denoising.
- Scope: one validated prompt pair, five targeted weights, top 20 raw LM-head scores, final prompt-conditioned canvas states, and no denoising.
