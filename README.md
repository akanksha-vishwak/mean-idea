# mean-idea

An experiment that averages two texts in a diffusion language model's
token-embedding space and asks the same model to denoise the result.

The reusable API has two model calls:

```python
left = model.text_to_embedding(text_a)
right = model.text_to_embedding(text_b)
result = model.embedding_to_text(mean_embeddings(left, right))
```

`DiffusionGemma` implements those calls with
`google/diffusiongemma-26B-A4B-it`. Each input is independently tokenized to
the model's 256-token canvas and run through its prompt-conditioned,
bidirectional diffusion decoder. Decoding projects the mean contextual hidden
states through the model's tied language-model head, then passes the resulting
token canvas as `decoder_input_ids` to DiffusionGemma's documented iterative
denoising loop.

This is contextual token-position interpolation, not a learned semantic latent
autoencoder. Tokenization differences can misalign otherwise related texts,
and the vocabulary projection loses information from the continuous mean.
Those limitations are part of the experiment rather than hidden by the API.

## Run

DiffusionGemma has 25.2B total parameters and needs a capable accelerator. The
first run downloads its weights from Hugging Face.

```console
uv sync
uv run mean-idea first.py second.py
uv run mean-idea first.py second.py --output result.py
```

Use `HF_TOKEN` if Hugging Face requires authentication. To install through the
Microsoft package feed proxy when public PyPI is blocked, use the normal
commands above; the proxy is configured as the project's default uv index.

The backend is configurable:

```console
uv run mean-idea first.py second.py --steps 24 \
  --prompt "Refine this canvas into one complete Python sorting program."
```

## Test

The unit tests do not download model weights:

```console
uv run pytest
```
