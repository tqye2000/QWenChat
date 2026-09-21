r"""
Lightweight web chat UI for Qwen3 via an OpenAI-compatible API endpoint.

This is the browser counterpart to `run_qwen.py`. It talks to the same local
inference server (e.g. the one started by `serve_qwen.py`) using the OpenAI
client, streams tokens live into a Gradio chat window, and surfaces any
reasoning/"thinking" output in a collapsible block above the final answer.

Environment variables (typically set in a local .env file):
  OPENAI_API_BASE   - base URL of your inference server
                      default: http://localhost:8080/v1
  OPENAI_API_KEY    - API key (any non-empty string for local servers)
                      default: "EMPTY"
  MODEL / OPENAI_MODEL - Hugging Face model id the server is serving
  GRADIO_SERVER_NAME   - interface to bind (default: 127.0.0.1)
  GRADIO_SERVER_PORT   - port for the UI (default: 7860)
  GRADIO_SHARE         - "1" to create a public share link (default: off)

Run (use the venv interpreter):
  .\.venv\Scripts\python.exe ui_qwen.py
"""

import os

import gradio as gr
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # read MODEL (and any overrides) from a local .env file
except ImportError:
    pass  # python-dotenv not installed; fall back to real environment vars

# ── Configuration (mirrors run_qwen.py) ────────────────────────────────────────
BASE_URL   = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
API_KEY    = os.getenv("OPENAI_API_KEY",  "EMPTY")
# Must be a real Hugging Face model id that your server can load/download.
MODEL_NAME = os.getenv("OPENAI_MODEL") or os.getenv("MODEL", "Qwen/Qwen3-32B")

SYSTEM_PROMPT    = "You are a helpful assistant."
MAX_TOKENS       = 4096
TEMPERATURE      = 1.0
TOP_P            = 0.95
PRESENCE_PENALTY = 1.5

# Attached-file context: which extensions to accept and how much text to inject
# (to avoid overflowing the model's context window). Oversized files are
# truncated with a marker.
TEXT_FILE_TYPES   = [".txt", ".md", ".markdown", ".csv", ".tsv", ".json",
                     ".yaml", ".yml", ".xml", ".log", ".py", ".js", ".ts",
                     ".html", ".css", ".ini", ".cfg", ".toml", ".rst"]
MAX_CONTEXT_CHARS = 100_000

# UI server settings.
UI_HOST  = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
UI_PORT  = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
UI_SHARE = os.getenv("GRADIO_SHARE", "0") == "1"
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def _read_text_file(path: str) -> str:
    """Read a text file as UTF-8 (replacing undecodable bytes), truncating if
    it exceeds MAX_CONTEXT_CHARS so a huge file can't blow the context window.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = fh.read(MAX_CONTEXT_CHARS + 1)
    if len(data) > MAX_CONTEXT_CHARS:
        data = data[:MAX_CONTEXT_CHARS] + "\n\n[... truncated ...]"
    return data


def _build_context(file_paths: list[str]) -> str:
    """Turn a list of uploaded file paths into a labelled context block.

    Each file's content is wrapped with its name so the model can tell where
    the attached context comes from. Unreadable files are reported inline.
    """
    blocks = []
    for path in file_paths:
        name = os.path.basename(path)
        try:
            content = _read_text_file(path)
        except OSError as exc:
            blocks.append(f"--- Attached file: {name} (could not be read: {exc}) ---")
            continue
        blocks.append(f"--- Attached file: {name} ---\n{content}\n--- End of {name} ---")
    return "\n\n".join(blocks)


def _split_message(message) -> tuple[str, list[str]]:
    """Normalise a ChatInterface message into (text, file_paths).

    With multimodal=True, `message` is a dict like {"text": str, "files": [...]}.
    For plain-string messages (or unexpected shapes) we degrade gracefully.
    """
    if isinstance(message, dict):
        text = message.get("text") or ""
        files = message.get("files") or []
        # Gradio may hand back dicts ({"path": ...}) or plain path strings.
        paths = [f.get("path") if isinstance(f, dict) else f for f in files]
        return text, [p for p in paths if p]
    return str(message or ""), []


def _render(reasoning: str, reply: str, done: bool = False) -> str:
    """Combine reasoning + answer into a single markdown string for the chat.

    Reasoning (if any) is wrapped in a collapsible <details> block so the final
    answer stays front and centre. While the model is still thinking (no answer
    text yet) the block is expanded; once the answer starts it collapses.
    """
    parts = []
    if reasoning.strip():
        open_attr = "" if (reply.strip() or done) else " open"
        parts.append(
            f"<details{open_attr}><summary>💭 Thinking</summary>\n\n"
            f"{reasoning.strip()}\n\n</details>"
        )
    if reply:
        parts.append(reply)
    return "\n\n".join(parts)


def respond(message, history: list[dict]):
    """Stream the assistant reply for a Gradio ChatInterface (multimodal).

    `message` is a dict {"text": str, "files": [paths]} because the textbox is
    multimodal. Any uploaded text files are read and prepended to the user turn
    as labelled context. `history` is a list of {"role", "content"} dicts; we
    prepend a system prompt, then stream the response, yielding progressively
    so tokens appear live in the browser.
    """
    text, file_paths = _split_message(message)
    context = _build_context(file_paths)
    if context:
        user_content = (
            "Use the following attached file(s) as context for my request.\n\n"
            f"{context}\n\n"
            + (f"My request:\n{text}" if text.strip() else "Please summarise the attached file(s).")
        )
    else:
        user_content = text

    if not user_content.strip():
        yield "_(Type a message or attach a text file to get started.)_"
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        # Skip non-text entries (e.g. gr.File payloads) which we don't resend.
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            presence_penalty=PRESENCE_PENALTY,
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any client/connection error
        yield (
            f"⚠️ Could not reach the model server at `{BASE_URL}`.\n\n"
            f"```\n{exc}\n```\n\n"
            "Make sure the inference server is running (e.g. "
            "`.\\.venv\\Scripts\\python.exe serve_qwen.py`) and that "
            "`OPENAI_API_BASE` points at it."
        )
        return

    reply = ""
    reasoning = ""
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Some servers (e.g. transformers serve) expose reasoning in a
            # separate 'reasoning_content' field instead of inline <think> tags.
            reasoning_token = getattr(delta, "reasoning_content", None)
            content_token = getattr(delta, "content", None)

            updated = False
            if reasoning_token:
                reasoning += reasoning_token
                updated = True
            if content_token:
                reply += content_token
                updated = True

            if updated:
                yield _render(reasoning, reply)
    except Exception as exc:  # noqa: BLE001 - connection dropped mid-stream
        yield _render(reasoning, reply) + f"\n\n⚠️ Stream interrupted: `{exc}`"
        return

    final = _render(reasoning, reply, done=True)
    if not final:
        if reasoning.strip():
            final = _render(reasoning, "", done=True) + (
                "\n\n_(The model produced only reasoning output with no final "
                "answer. Try QWEN_REASONING=off to disable thinking mode.)_"
            )
        else:
            final = "_(The model returned an empty response.)_"
    yield final


def build_ui() -> gr.ChatInterface:
    """Create the Gradio chat interface."""
    description = (
        f"Model: `{MODEL_NAME}`  •  Server: `{BASE_URL}`\n\n"
        "Attach a text file (paperclip) to use its contents as context for your prompt."
    )
    return gr.ChatInterface(
        fn=respond,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_types=TEXT_FILE_TYPES,
            file_count="multiple",
            placeholder="Ask something, or attach a text file as context…",
        ),
        title="Qwen Chat",
        description=description,
        examples=[
            {"text": "Explain quantum entanglement in simple terms."},
            {"text": "Write a haiku about local LLMs."},
            {"text": "What's the difference between a list and a tuple in Python?"},
        ],
    )


def main():
    ui = build_ui()
    ui.launch(server_name=UI_HOST, server_port=UI_PORT, share=UI_SHARE)


if __name__ == "__main__":
    main()
