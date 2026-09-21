"""Download a Hugging Face model snapshot into the local cache.

Trusts the Windows certificate store (for corporate TLS-intercepting proxies)
via truststore, and explicitly runs ONLINE (it clears the offline env vars that
`serve_qwen.py` sets) so the files are actually fetched.

Usage:
    ./.venv/Scripts/python.exe download_model.py                    # QWEN_MODEL or default
    ./.venv/Scripts/python.exe download_model.py Qwen/Qwen3-8B
    ./.venv/Scripts/python.exe download_model.py Qwen/Qwen3-8B --revision main

Env vars:
    QWEN_MODEL   - default model id if none is passed on the command line
    HF_TOKEN     - Hugging Face access token (only needed for gated/private repos)
    HF_ENDPOINT  - alternative Hub mirror, e.g. https://hf-mirror.com
"""
import argparse
import os
import sys

# A download must NOT run in offline mode. Clear anything inherited from the
# shell / serve_qwen.py before huggingface_hub is imported.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

try:
    from dotenv import load_dotenv
    load_dotenv()  # read MODEL (and any overrides) from a local .env file
except ImportError:
    pass  # python-dotenv not installed; fall back to real environment vars

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass  # pip-system-certs may already inject the system certificates

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model into the local cache."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=os.environ.get("QWEN_MODEL") or os.environ.get("MODEL", "Qwen/Qwen3-8B"),
        help="Model id, e.g. Qwen/Qwen3-8B (default: %(default)s)",
    )
    parser.add_argument("--revision", default="main", help="Branch/tag/commit (default: main)")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token for gated/private repos (default: $HF_TOKEN)",
    )
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"Endpoint : {endpoint}")
    print(f"Model    : {args.model}@{args.revision}")
    print("Downloading (this can take a while for multi-GB models)...\n")

    try:
        path = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            token=args.token,
            # Weights + tokenizer/config are enough for transformers serving;
            # skip large alternative formats we don't use.
            ignore_patterns=["*.gguf", "*.onnx", "*.pth", "original/*", "coreml/*"],
        )
    except GatedRepoError:
        print(f"\n[Gated repo] {args.model} requires accepting its license and a token.")
        print("Accept the license on the model page, then set HF_TOKEN and retry.")
        return 1
    except RepositoryNotFoundError:
        print(f"\n[Not found] '{args.model}' does not exist (typo?) or needs a token.")
        return 1
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", "?")
        print(f"\n[HTTP {status}] {exc}")
        if str(status) == "403":
            print(
                "\nA 403 here means your corporate proxy is blocking the Hugging Face\n"
                "Hub API (the failing URL often has a '?_sm_nck=1' proxy marker). Options:\n"
                "  1. Set a permitted mirror:  $env:HF_ENDPOINT = 'https://hf-mirror.com'\n"
                "  2. Download on a network that allows huggingface.co, then copy the\n"
                "     %USERPROFILE%\\.cache\\huggingface\\hub folder onto this machine.\n"
                "  3. Ask IT to allow-list huggingface.co and *.hf.co."
            )
        return 1

    print(f"\nDone. Snapshot cached at:\n  {path}")
    print("\nServe it with:")
    print(f'  $env:QWEN_MODEL = "{args.model}"; .\\.venv\\Scripts\\python.exe serve_qwen.py')
    return 0


if __name__ == "__main__":
    sys.exit(main())
