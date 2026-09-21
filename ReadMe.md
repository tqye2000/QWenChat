# Testing a locally hosted Qwen3 model

This project runs a Qwen3 model locally via `transformers serve` (an
OpenAI-compatible API server) and chats with it using `run_qwen.py`.

## Setup

Create a virtual environment and install all dependencies (including
GPU-enabled PyTorch with CUDA 12.8 wheels):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> If you're behind a corporate TLS-intercepting proxy, `truststore` +
> `pip-system-certs` let Python trust the Windows certificate store instead of
> failing with `CERTIFICATE_VERIFY_FAILED`. Skip this if you don't need it.

The CUDA 12.8 wheels are compatible with CUDA 13.x drivers. Requires 4×
NVIDIA RTX 6000 Ada (or any GPU with ≥ the required VRAM below).

## Running the server

Use `serve_qwen.py` instead of calling `transformers serve` directly — it
applies the certificate fix and picks the right CLI flags for this
`transformers` version.

```powershell
.\.venv\Scripts\python.exe serve_qwen.py
```

Env vars (all optional):

| Variable         | Default          | Purpose                                                                 |
|------------------|------------------|--------------------------------------------------------------------------|
| `QWEN_MODEL`     | `Qwen/Qwen3-32B` | Hugging Face model id to serve                                           |
| `QWEN_MULTI_GPU` | `1`              | Set to `1` to shard a large model across all visible GPUs (accelerate). Disables `--continuous-batching` automatically. |
| `QWEN_REASONING` | `auto`           | Reasoning mode for supported models: `on`, `off`, or `auto`.            |

Examples:

```powershell
# Default: Qwen3-8B on a single GPU with continuous batching
.\.venv\Scripts\python.exe serve_qwen.py

# A bigger model, single GPU
$env:QWEN_MODEL = "Qwen/Qwen3-14B"
.\.venv\Scripts\python.exe serve_qwen.py

# A model too large for one GPU, sharded across all available GPUs
$env:QWEN_MODEL = "Qwen/Qwen3-32B"
$env:QWEN_MULTI_GPU = "1"
.\.venv\Scripts\python.exe serve_qwen.py

# Force visible reasoning tokens (for models/templates that expose them)
$env:QWEN_REASONING = "on"
.\.venv\Scripts\python.exe serve_qwen.py
```

The server listens on `http://localhost:8000` and lazily downloads/loads the
model on the first request.

### Equivalent plain CLI (single GPU, no cert fix)

If you don't need the certificate fix or env-var switches, the underlying
command is:

```powershell
transformers serve --port 8000 --continuous-batching Qwen/Qwen3-8B
```

## Chatting with the model

In another terminal, run the interactive chat client:

```powershell
.\.venv\Scripts\python.exe run_qwen.py
```

Env vars (all optional):

| Variable          | Default                     | Purpose                              |
|--------------------|-----------------------------|---------------------------------------|
| `OPENAI_API_BASE`  | `http://localhost:8000/v1`  | Base URL of the inference server      |
| `OPENAI_API_KEY`   | `EMPTY`                     | API key (any non-empty string works)  |
| `OPENAI_MODEL`     | `Qwen/Qwen3-8B`              | Model id to request (must match what the server is serving) |
| `OPENAI_REASONING` | `auto`                      | Per-request reasoning mode: `on`, `off`, or `auto`. |

Type `quit` or `exit` to stop the chat.

## Choosing a model (VRAM guide, bf16)

| Model                    | Approx. VRAM | Fits single 48GB GPU? |
|--------------------------|-------------:|------------------------|
| `Qwen/Qwen3-0.6B`        | ~1.5 GB      | Yes                    |
| `Qwen/Qwen3-1.7B`        | ~4 GB        | Yes                    |
| `Qwen/Qwen3-4B`          | ~9 GB        | Yes                    |
| `Qwen/Qwen3-8B`          | ~17 GB       | Yes                    |
| `Qwen/Qwen3-14B`         | ~29 GB       | Yes                    |
| `Qwen/Qwen3-30B-A3B` (MoE) | ~62 GB    | No — use `QWEN_MULTI_GPU=1` |
| `Qwen/Qwen3-32B`         | ~65 GB       | No — use `QWEN_MULTI_GPU=1` |
| `Qwen/Qwen3-235B-A22B` (MoE) | ~470 GB  | No — needs true tensor-parallel serving (e.g. vLLM), not supported here |

All Qwen3 models are reasoning models and may emit `<think>...</think>`
blocks before the final answer.