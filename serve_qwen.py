"""Launch `transformers serve` with corporate-CA trust enabled.

Run with the venv interpreter so the Windows certificate store is trusted:
    ./.venv/Scripts/python.exe serve_qwen.py

On startup it force-restarts: any process already listening on the server port
is terminated first so a fresh server can bind.

Env vars:
  QWEN_MODEL      - HF model id to serve (default: Qwen/Qwen3-32B)
  QWEN_PORT       - Port to serve on (default: 8000). Any existing listener on
                    this port is killed before starting.
  QWEN_MULTI_GPU  - "1" (default) to shard a large model across all visible GPUs
                    via accelerate (device_map=auto). Continuous batching does
                    not support sharded models, so it is disabled automatically
                    in this mode (generation is still correct, just without
                    the continuous-batching throughput optimization). Set to "0"
                    for a small model that fits on a single GPU.
  QWEN_OFFLINE    - Force offline ("1") or online ("0"). If unset, the script
                    auto-detects: OFFLINE when the model is already cached,
                    ONLINE (to download on first request) when it is not.
                    To pre-download a model, run: download_model.py <model-id>.
  QWEN_ATTN       - Attention implementation to request (e.g. flash_attention_2,
                    sdpa, eager). Default: "flash_attention_2" when the
                    `flash_attn` package is installed, otherwise the flag is
                    omitted and transformers picks its own default (sdpa). Set
                    to "0"/"none" to never pass the flag, or to any explicit
                    value to force it. flash_attention_2 lowers KV-cache/
                    activation memory and speeds up long contexts.
    QWEN_REASONING  - Reasoning mode for models that support it: on/off/auto.
                                        Default: auto.
"""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # read MODEL (and any overrides) from a local .env file
except ImportError:
    pass  # python-dotenv not installed; fall back to real environment vars

#MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-1.7B")
#MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")
# MODEL comes from .env (MODEL=...). QWEN_MODEL still overrides if set.
MODEL = os.environ.get("QWEN_MODEL") or os.environ.get("MODEL", "Qwen/Qwen3-32B")
# Qwen3-32B (~65 GB bf16) does not fit on one 48 GB GPU, so shard it across all
# visible GPUs by default. Continuous batching is disabled automatically in this
# mode (see below). Override with QWEN_MULTI_GPU=0 for a single-GPU model.
MULTI_GPU = os.environ.get("QWEN_MULTI_GPU", "1") == "1"
PORT = int(os.environ.get("QWEN_PORT", "8000"))
# Attention backend. Unset -> auto: use flash_attention_2 if available, else let
# transformers default (sdpa). "0"/"none"/"" -> never pass the flag.
ATTN_IMPL = os.environ.get("QWEN_ATTN")
# Reasoning mode. Unset -> auto. Accepts on/off/auto.
REASONING = os.environ.get("QWEN_REASONING", "auto")


def _flash_attn_available() -> bool:
    """True if the `flash_attn` package can be imported in this environment."""
    from importlib.util import find_spec

    try:
        return find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def _resolve_attn_impl() -> str | None:
    """Decide which attn_implementation to request, or None to omit the flag.

    * QWEN_ATTN unset  -> flash_attention_2 if `flash_attn` is installed, else
      None (transformers picks sdpa).
    * QWEN_ATTN in {"0","none",""} -> None (never pass the flag).
    * QWEN_ATTN=<value> -> use <value> verbatim, but if it is
      flash_attention_2 and `flash_attn` is missing, warn and fall back to None.
    """
    if ATTN_IMPL is None:
        if _flash_attn_available():
            print("[serve_qwen] flash_attn detected -> using --attn_implementation flash_attention_2.")
            return "flash_attention_2"
        print("[serve_qwen] flash_attn not installed -> using transformers default attention (sdpa).")
        return None
    choice = ATTN_IMPL.strip().lower()
    if choice in {"0", "none", ""}:
        return None
    if choice == "flash_attention_2" and not _flash_attn_available():
        print("[serve_qwen] QWEN_ATTN=flash_attention_2 requested but `flash_attn` is not installed "
              "-> falling back to transformers default (sdpa). Install with: pip install flash-attn")
        return None
    print(f"[serve_qwen] using --attn_implementation {ATTN_IMPL.strip()}.")
    return ATTN_IMPL.strip()


def _resolve_reasoning() -> str:
    """Validate QWEN_REASONING and return one of: on, off, auto."""
    choice = REASONING.strip().lower()
    if choice in {"on", "off", "auto"}:
        print(f"[serve_qwen] using --reasoning {choice}.")
        return choice
    print(f"[serve_qwen] invalid QWEN_REASONING='{REASONING}' -> using auto.")
    return "auto"


def _free_port(port: int) -> None:
    """Kill any process already listening on `port` so we can (re)bind it.

    `transformers serve` fails to start if the port is taken (e.g. a previous
    instance is still running). This force-restarts by terminating the old
    listener(s) first. The current process is never a listener yet, so it is
    safe from being killed here.
    """
    import psutil

    me = os.getpid()
    killed = []
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            pid = conn.pid
            if pid is None or pid == me:
                continue
            try:
                proc = psutil.Process(pid)
                name = proc.name()
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed.append((pid, name))
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                print(f"[serve_qwen] could not stop PID {pid} on port {port}: {exc}")
    for pid, name in killed:
        print(f"[serve_qwen] stopped existing server on port {port} (PID {pid}, {name}).")
    if not killed:
        print(f"[serve_qwen] port {port} is free.")


def _hub_cache_dir() -> Path:
    """Location of the Hugging Face hub cache, honoring HF_HUB_CACHE / HF_HOME."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _is_cached(model_id: str) -> bool:
    """True if a usable snapshot of `model_id` exists locally.

    This is a pure filesystem check on purpose: it must NOT import
    `huggingface_hub`, because importing that module locks its
    `HF_HUB_OFFLINE` constant from the environment. We have to decide offline
    vs online and set the env vars BEFORE huggingface_hub (and transformers,
    which derives its offline state from it) is imported anywhere. Otherwise
    the offline switch is ignored and `transformers serve` makes a Hub API
    call that a corporate proxy blocks with 403 -> opaque 500 on every request.
    """
    repo = "models--" + model_id.replace("/", "--")
    repo_dir = _hub_cache_dir() / repo
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    candidates = []
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text(encoding="utf-8").strip()
        candidates.append(snapshots / commit)
    candidates += [p for p in snapshots.iterdir() if p.is_dir()]
    for snap in candidates:
        if (snap / "config.json").is_file() and any(snap.glob("*.safetensors")):
            return True
    return False


# Decide online vs offline:
#   * If the model is already cached -> force OFFLINE. Behind a TLS-intercepting
#     corporate proxy the HF Hub API returns HTTP 403, which `transformers serve`
#     turns into an opaque 500 on every request even for cached models. Offline
#     mode skips that API call entirely.
#   * If the model is NOT cached -> stay ONLINE so the server can download it on
#     first use (requires the proxy to permit huggingface.co; otherwise run
#     download_model.py first).
# Override explicitly with QWEN_OFFLINE=1 (force offline) or QWEN_OFFLINE=0.
_offline_override = os.environ.get("QWEN_OFFLINE")
if _offline_override is not None:
    OFFLINE = _offline_override == "1"
else:
    OFFLINE = _is_cached(MODEL)

if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[serve_qwen] '{MODEL}' found in cache -> running OFFLINE.")
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    print(f"[serve_qwen] '{MODEL}' not fully cached -> running ONLINE to download it.")

if not MULTI_GPU:
    # Force a single GPU so accelerate does not shard the model across
    # multiple devices (continuous batching cannot handle a split model).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass  # pip-system-certs may already inject at startup

from transformers.cli.transformers import main

if __name__ == "__main__":
    # Force-restart: free the port if a previous server is still running.
    _free_port(PORT)

    sys.argv = [
        "transformers", "serve",
        "--port", str(PORT),
        MODEL,
    ]
    if MULTI_GPU:
        sys.argv += ["--device", "auto"]
    else:
        sys.argv += ["--continuous-batching"]

    attn_impl = _resolve_attn_impl()
    if attn_impl is not None:
        sys.argv += ["--attn-implementation", attn_impl]
    sys.argv += ["--reasoning", _resolve_reasoning()]
    main()
