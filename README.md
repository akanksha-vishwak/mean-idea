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

The default is a 4.27M-parameter random DiffusionGemma fixture for local plumbing
tests. It exercises the compatible model API but does not produce useful text.

```console
uv sync
uv run mean-idea first.py second.py
uv run mean-idea first.py second.py --output result.py
```

DiffusionGemma's trained canvas remains fixed at 256 tokens. Process longer
documents as sequential canvases by selecting a fixed input length shared by
both files:

```console
uv run mean-idea first.py second.py --max-input-tokens 4096
```

The value must be a multiple of 256. Each canvas is conditioned on the
preceding chunks, and the resulting hidden-state matrices are concatenated
before interpolation. The upper bound is the model's 262,144-token context
minus the instruction prompt, rounded down to a whole canvas. Large values
require one model pass per input canvas and one diffusion decode per output
canvas, so CPU execution time grows substantially.

On Linux x86-64, the lock file selects the official PyTorch CUDA 12.8 wheels
instead of whichever PyTorch build is newest on the default package index.
Those wheels bundle the CUDA runtime; the host only needs a compatible NVIDIA
driver. Confirm that the environment can see the GPU before downloading a
large model:

```console
uv run python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Select the useful 25.2B BF16 reference model explicitly. Its weights need about
51 GB, plus runtime memory:

```console
uv run mean-idea first.py second.py \
  --model-id google/diffusiongemma-26B-A4B-it
```

There is currently no useful 2-4 GB DiffusionGemma checkpoint compatible with
this Transformers adapter. For a GPU with 32 GB RAM, the 16.8 GB
`unsloth/diffusiongemma-26B-A4B-it-GGUF` Q4_K_M checkpoint fits, but it requires
llama.cpp's dedicated diffusion runner and cannot be passed to `--model-id`.

Use `HF_TOKEN` if Hugging Face requires authentication. To install through the
Microsoft package feed proxy when public PyPI is blocked, use the normal
commands above; the proxy is configured as the project's default uv index.

The backend is configurable:

```console
uv run mean-idea first.py second.py --steps 24 \
  --prompt "Refine this canvas into one complete Python sorting program."
```

## Remote model

Run the model on a GPU host while keeping interpolation on that host. The
server exposes `/encode` and `/decode` for diagnostics and `/interpolate` for
normal use. The latter avoids transferring the contextual embedding matrices
over the network.

```console
uv sync --extra server
MEAN_IDEA_MODEL_ID=google/diffusiongemma-26B-A4B-it \
MEAN_IDEA_MAX_INPUT_TOKENS=4096 \
  uv run --extra server uvicorn mean_idea.server:app --host 0.0.0.0 --port 8000
```

Point the CLI at the service. Set `MEAN_IDEA_API_TOKEN` when the hosting
platform expects bearer authentication:

```console
uv run mean-idea first.py second.py \
  --backend remote \
  --endpoint-url https://example.endpoints.huggingface.cloud \
  --model-id google/diffusiongemma-26B-A4B-it
```

The client and server reject mismatched protocol versions and model IDs before
processing embeddings. The remote service currently relies on the hosting
platform for authentication and TLS termination.

## Test

The unit tests do not download model weights:

```console
uv run pytest
```
