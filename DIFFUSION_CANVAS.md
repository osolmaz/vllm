# Diffusion canvas streaming fork

This fork adds an opt-in side channel that streams the intermediate canvas
(the partially denoised token block) of diffusion LLMs such as DiffusionGemma
on every denoising step, so clients can visualize how the text actually forms.
The OpenAI-compatible completion stream is unchanged, and with the flag off
(the default) behavior and wire format are identical to upstream vLLM.

Upstream base: `4e5ca89cfe98121642d76b40e32a006f4d0fbf3b` (v0.23.1 development
line). All changes are pure Python; no CUDA kernels are modified.

## Install

The official precompiled kernels for the upstream base commit are reused, so
no compilation happens:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=4e5ca89cfe98121642d76b40e32a006f4d0fbf3b \
pip install git+https://github.com/osolmaz/vllm@canvas-v0.23.1rc3
```

`VLLM_PRECOMPILED_WHEEL_COMMIT` must stay in sync with the upstream base
commit above; it pins which official wheel provides the binary artifacts.

## Serve

Any diffusion model works. The command used for development and demos:

```bash
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --diffusion-stream-canvas
```

## The side channel

`GET /v1/diffusion/events` is a server-sent-events stream. Each event is a
JSON object:

```json
{"request_id": "chatcmpl-...", "step": 12, "block": 0, "text": "partially denoised canvas ..."}
```

- `text` is the detokenized canvas scheduled for one denoising step: accepted
  tokens mixed with the sampler's renoise tokens. Unfilled positions render
  as `░`.
- `step` counts denoising steps observed for the request (monotonic across
  blocks, not per block).
- `block` is the commit ordinal of the block the snapshot belongs to (the
  number of commits the request has streamed so far). The canvas feed and
  the completion stream travel on separate connections, so clients should
  discard snapshots whose `block` is at or below the ordinal of a commit
  they have already received; rendering one would duplicate committed text.
- Pass `?request_id=<id>` to scope the stream to a single request; clients
  that set the `X-Request-Id` header on their completion request know the id
  (`chatcmpl-<header>`) up front and should subscribe scoped. An unscoped
  subscription streams every request on the server and is intended for
  server-side observability.

The flag requires the single-process FastAPI frontend (no `--grpc`, headless
mode, or external/hybrid/multi-port data-parallel load balancing), because
the event broadcaster is process-local.

## Client

The [localpi](https://github.com/osolmaz/localpi) `--diffusion-canvas`
flag renders these events live in the Pi TUI; the widget is also available
as a standalone Pi package from the same repository.

## Branch maintenance

The feature lives on the `diffusion-canvas-events` branch; releases are tags
named `canvas-v<upstream version>rc<N>` (PEP 440-parseable so vLLM's version detection accepts them; the installed version reports the tag). Rebasing onto a newer upstream base means
re-running the test suite (`tests/v1/engine/test_diffusion_events.py`),
updating the base commit here, and cutting the next tag.
