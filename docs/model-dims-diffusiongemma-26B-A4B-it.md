# DiffusionGemma 26B A4B IT: architecture and tensor dimensions

This document describes the Hugging Face checkpoint
[`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
and how `mean-idea` uses it. Dimensions are taken from the checkpoint's
[`config.json`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/config.json)
and
[`generation_config.json`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/generation_config.json).

## Dimension notation

| Symbol | Meaning | Default |
|---|---|---:|
| `B` | Batch size | Application uses `1` |
| `P` | Encoded prompt/context length | Variable |
| `C` | Diffusion canvas length | `256` tokens |
| `M` | Number of sequential canvases | Application default is `1` |
| `L` | Application input length, `C * M` | `256` tokens |
| `H` | Text hidden size | `2,816` |
| `V` | Vocabulary size | `262,144` |
| `E` | Number of routed experts | `128` |
| `K` | Experts selected per token | `8` |

All sizes below omit the batch dimension when `B = 1`.

## High-level architecture

DiffusionGemma is a sparse Mixture-of-Experts transformer with two execution
roles:

1. An **autoregressive encoder** processes the prompt and completed earlier
   canvases. It writes their keys and values into a KV cache.
2. A **bidirectional diffusion decoder** processes every position in the
   current canvas in parallel. Its attention can read both the encoder KV cache
   and all positions in the current canvas.

The decoder does not receive a conventional encoder hidden-state tensor through
a separate cross-attention layer. Instead, encoder context is exposed as a
read-only KV cache to the decoder's attention layers.

The encoder and decoder share the text transformer weights, including token
embeddings, attention, MLP, MoE, and final normalization weights. The decoder
adds a self-conditioning module used by iterative diffusion. The language-model
head is tied to the token embedding table.

Official headline values:

| Property | Value |
|---|---:|
| Total parameters | 25.2B |
| Active parameters per token | 3.8B |
| Text transformer layers | 30 |
| Routed experts | 128 |
| Active routed experts per token | 8 |
| Shared expert/MLP paths | 1 |
| Default canvas | 256 tokens |
| Maximum configured positions | 262,144 |
| Sliding-attention window | 1,024 tokens |
| Vision encoder | Approximately 550M parameters |
| Weight dtype | BF16 |

The model card describes the context limit as "up to 256K"; the exact
configuration value is `262,144`.

## Text transformer dimensions

### Core dimensions

| Configuration field | Value |
|---|---:|
| Hidden size | `2,816` |
| Layers | `30` |
| Vocabulary | `262,144` |
| Query heads | `16` |
| Sliding-attention KV heads | `8` |
| Sliding-attention head dimension | `256` |
| Full-attention KV heads | `2` |
| Full-attention head dimension | `512` |
| Shared MLP intermediate size | `2,112` |
| Routed-expert intermediate size | `704` |
| Routed experts | `128` |
| Top-k routed experts | `8` |
| RMSNorm epsilon | `1e-6` |
| Logit soft cap | `30.0` |

The 30 layers follow a repeating pattern of five sliding-attention layers and
one full-attention layer:

```text
[sliding, sliding, sliding, sliding, sliding, full] x 5
```

### Attention projections

The hidden input and output of every attention block remain `[B, S, 2816]`,
but query/key/value projections use wider or narrower internal dimensions.

#### Sliding-attention layer

| Tensor | Shape before sequence axes |
|---|---:|
| Query | `16 heads * 256 = 4,096` |
| Key | `8 heads * 256 = 2,048` |
| Value | `8 heads * 256 = 2,048` |
| Attention output projection | `4,096 -> 2,816` |

The eight KV heads are repeated across the 16 query heads, giving two query
heads per KV head. Attention is limited to a 1,024-token sliding window.

#### Full-attention layer

| Tensor | Shape before sequence axes |
|---|---:|
| Query | `16 heads * 512 = 8,192` |
| Key | `2 heads * 512 = 1,024` |
| Value | `2 heads * 512 = 1,024` |
| Attention output projection | `8,192 -> 2,816` |

The two KV heads are repeated across 16 query heads, giving eight query heads
per KV head. In full-attention layers, values reuse the projected key tensor
before separate key and value normalization.

The encoder attention is causal for text. The diffusion decoder attention is
non-causal within the current canvas.

### Shared MLP and routed MoE

Every text layer contains both:

- A shared gated MLP with intermediate width `2,112`.
- A sparse MoE path with 128 experts, each with intermediate width `704`.

For a token representation `[2816]`, the shared gated MLP is:

```text
gate: 2816 -> 2112
up:   2816 -> 2112
SiLU(gate) * up
down: 2112 -> 2816
```

The router maps `[B*S, 2816]` to `[B*S, 128]`, applies a softmax, chooses the
top eight experts, and renormalizes their weights.

Each routed expert is:

```text
gate: 2816 -> 704
up:   2816 -> 704
SiLU(gate) * up
down: 704 -> 2816
```

The implementation stores all expert weights compactly as:

| Parameter | Shape |
|---|---:|
| Combined expert gate/up | `[128, 1408, 2816]` |
| Expert down projection | `[128, 2816, 704]` |

Only eight experts execute for each token, but weights for all 128 experts must
be stored.

## Token embeddings and language-model output

The tied token embedding / language-model-head matrix has shape:

```text
[V, H] = [262144, 2816]
```

It contains `738,197,504` parameters:

| Dtype | Storage |
|---|---:|
| BF16 | 1,476,395,008 bytes (about 1.375 GiB) |
| FP32 | 2,952,790,016 bytes (about 2.75 GiB) |

Token IDs `[B, S]` are looked up in this matrix to produce token embeddings
`[B, S, 2816]`. After the final transformer layer, the same matrix is used as
the output projection:

```text
[B, C, 2816] @ [2816, 262144] -> [B, C, 262144]
```

The output logits are explicitly converted to FP32 and soft-capped to the
interval `[-30, 30]`.

## Diffusion decoder inputs and outputs

For one denoising step:

| Value | Shape | Meaning |
|---|---:|---|
| Encoder token IDs | `[B, P]` | New prompt/context tokens |
| Encoder attention mask | `[B, P]` | Valid context positions |
| Canvas token IDs | `[B, C]` | Current noisy or partially denoised canvas |
| Canvas position IDs | `[B, C]` | Positions following encoded context |
| Previous self-conditioning logits | `[B, C, V]` | Optional previous-step predictions |
| Decoder hidden state | `[B, C, H]` | Contextual canvas representation |
| Output logits | `[B, C, V]` | Score for every vocabulary item at every canvas position |

On the first denoising step there are no self-conditioning logits, so the
self-conditioning signal is zero. On later steps:

```text
softmax([B, C, V]) @ embedding_table[V, H]
    -> soft embedding [B, C, H]
```

That soft embedding passes through a gated `2816 -> 2112 -> 2816`
self-conditioning MLP and is combined with the current canvas token
embeddings.

The default generation configuration uses:

| Setting | Value |
|---|---:|
| Maximum denoising steps | `48` |
| Temperature schedule | `0.8 -> 0.4` |
| Entropy bound | `0.1` |
| Confidence threshold | `0.005` |
| Stability threshold | `1` |
| Default generated canvas | `256` tokens |

The entropy-bound sampler accepts the lowest-entropy positions and replaces
the remaining positions with new random vocabulary IDs for another denoising
step. A completed canvas is appended to the encoder context before generating
the next canvas.

## Why canvas length consumes so much memory

Canvas length is a token count, but logits have one vocabulary-sized row for
every canvas token:

```text
logit elements = B * C * V
```

For batch size one:

| Canvas | Logit elements | BF16 | FP32 |
|---:|---:|---:|---:|
| 256 | 67,108,864 | 128 MiB | 256 MiB |
| 2,048 | 536,870,912 | 1 GiB | 2 GiB |

This is why a 2,048-token canvas can trigger a 2 GiB allocation even though
`--canvas-length` is measured in tokens. Temporary probability, entropy, and
self-conditioning operations can require additional storage.

The checkpoint was trained and published with a 256-token canvas. Transformers
allows `config.canvas_length` to be changed, but quality at other canvas sizes
is not guaranteed by the model card.

## Vision input dimensions

The vision tower is a 27-layer Gemma 4 vision transformer:

| Property | Value |
|---|---:|
| Vision hidden size | `1,152` |
| Layers | `27` |
| Attention heads | `16` |
| KV heads | `16` |
| Head dimension | `72` |
| MLP intermediate size | `4,304` |
| Patch size | `16 x 16` pixels |
| Patch input width | `16 * 16 * 3 = 768` |
| Pooling kernel | `3 x 3` patches |
| Default image soft tokens | `280` |

The processor converts RGB images into patch vectors:

```text
pixel patches: [B, Npatch, 768]
patch projection: 768 -> 1152
vision states: [B, Npatch, 1152]
3x3 spatial pooling: 9 patches -> 1 soft token
language projection: 1152 -> 2816
```

The projected soft image tokens replace image-placeholder positions in the
text input stream, so the text transformer ultimately sees both text and image
tokens as width-2,816 vectors.

The default processor requests 280 image soft tokens. The model card also
documents supported budgets of 70, 140, 280, 560, and 1,120 soft tokens.

## What "embedding" means in `mean-idea`

The application does **not** use a pooled sentence embedding and does not
extract the static token lookup vectors directly.

For each input text and each canvas, `mean-idea`:

1. Tokenizes and pads/truncates the text to `L = C * M` token IDs.
2. Sends earlier input canvases through the encoder as context.
3. Sends the current input canvas as `decoder_input_ids`.
4. Extracts the diffusion decoder's final contextual hidden state.

For one canvas the model returns:

```text
last_hidden_state: [1, C, 2816]
```

The batch dimension is removed and the values are copied to CPU FP32:

```text
TextEmbedding.values: [C, 2816]
```

With `M` sequential canvases, their hidden states are concatenated:

```text
TextEmbedding.values: [C * M, 2816]
```

Examples:

| CLI configuration | Application embedding shape | FP32 storage |
|---|---:|---:|
| `--canvas-length 256` | `[256, 2816]` | 2,883,584 bytes (about 2.75 MiB) |
| `--canvas-length 2048` | `[2048, 2816]` | 23,068,672 bytes (about 22 MiB) |
| `--canvas-length 256 --multi-canvas 4` | `[1024, 2816]` | 11,534,336 bytes (about 11 MiB) |

This representation is:

- Contextual: each row depends on the prompt and attention context.
- Position-preserving: there is one 2,816-value vector per token position.
- Fixed-length for a selected `C` and `M`.
- Not normalized or pooled into one vector.
- Not a learned semantic bottleneck or sentence embedding.

Two application embeddings must have identical shapes. `mean-idea` converts
them to FP32, stacks them as `[N, L, 2816]`, and takes an elementwise mean over
`N`:

```text
mean embedding: [L, 2816]
```

## How `mean-idea` turns the mean embedding back into text

For each row of the mean contextual embedding, the application finds the
highest-scoring vocabulary entry using the tied LM-head matrix:

```text
[L, 2816] @ [2816, 262144] -> conceptual logits [L, 262144]
argmax over vocabulary -> initial token IDs [L]
```

The implementation chunks the vocabulary projection so it does not retain the
entire `[L, 262144]` matrix. The resulting token IDs are only an initialization
for generation. Each canvas is then passed to DiffusionGemma's iterative
denoising loop, conditioned on the generation prompt and any completed earlier
canvases.

This projection is lossy: arbitrary continuous hidden vectors are not generally
equal to token embedding rows, and choosing the nearest vocabulary direction
discards information before diffusion refinement.

## End-to-end shape summary

For the application's default `B=1`, `C=256`, and `M=1`:

```text
source text
  -> tokenizer IDs                         [1, 256]
  -> decoder contextual hidden states      [1, 256, 2816]
  -> application TextEmbedding             [256, 2816] FP32

two TextEmbeddings
  -> stack                                 [2, 256, 2816]
  -> mean                                  [256, 2816]
  -> tied-head vocabulary projection       [256, 262144] conceptually
  -> argmax token canvas                   [1, 256]
  -> iterative diffusion logits            [1, 256, 262144] FP32
  -> generated token IDs                   [1, <=256]
  -> decoded output string
```

For `--canvas-length 2048`, replace every canvas axis of 256 with 2,048.
The hidden-state representation grows linearly and remains relatively small;
the vocabulary logits also grow linearly but have width 262,144, making them
the dominant transient allocation.

## Sources

- [Official Hugging Face model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [Checkpoint configuration](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/config.json)
- [Generation configuration](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/generation_config.json)
- [Processor configuration](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/processor_config.json)
- [Transformers DiffusionGemma implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/diffusion_gemma)
- [`mean-idea` DiffusionGemma adapter](../src/mean_idea/diffusion_gemma.py)
