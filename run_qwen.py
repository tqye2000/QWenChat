"""
Interactive chat script for Qwen3.6 via an OpenAI-compatible API endpoint.

The Qwen3.6 model is served by a local (or remote) inference server such as:
  - vLLM   : vllm serve Qwen/Qwen3.6-27B-GPTQ-Int4 --port 8080
  - Ollama : ollama run qwen3.6
  - LM Studio, llama.cpp (with OpenAI-compat mode), etc.

Environment variables (set before running):
  OPENAI_API_BASE   – base URL of your inference server
                      default: http://localhost:8080/v1
  OPENAI_API_KEY    – API key (can be any non-empty string for local servers)
                      default: "EMPTY"

Requirements:
  pip install openai
"""

import os
import sys
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # read MODEL (and any overrides) from a local .env file
except ImportError:
    pass  # python-dotenv not installed; fall back to real environment vars

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL   = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
API_KEY    = os.getenv("OPENAI_API_KEY",  "EMPTY")
# Must be a real Hugging Face model id that your server can load/download.
# Set it once in .env (MODEL=...). OPENAI_MODEL still overrides if set.
#MODEL_NAME = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-1.7B")
#MODEL_NAME = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
MODEL_NAME = os.getenv("OPENAI_MODEL") or os.getenv("MODEL", "Qwen/Qwen3-32B")

SYSTEM_PROMPT    = "You are a helpful assistant."
MAX_TOKENS       = 2048
TEMPERATURE      = 1.0
TOP_P            = 0.95
PRESENCE_PENALTY = 1.5
# ──────────────────────────────────────────────────────────────────────────────


def create_client() -> OpenAI:
    """Create an OpenAI client pointed at the local inference server."""
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


def chat_once(client: OpenAI, messages: list[dict]) -> str:
    """Stream the reply for the conversation history, printing tokens live.

    `transformers serve` always responds with a streaming (SSE) body, so we must
    request stream=True and accumulate the delta chunks.
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        stream=True,
    )
    reply = ""
    thinking = ""
    in_thinking = False
    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        # Some servers (e.g. transformers serve) expose reasoning in a
        # separate 'reasoning_content' field instead of inline <think> tags.
        reasoning_token = getattr(delta, "reasoning_content", None)
        content_token = delta.content

        if reasoning_token:
            if not in_thinking:
                print("<think>", end="", flush=True)
                in_thinking = True
            print(reasoning_token, end="", flush=True)
            thinking += reasoning_token
        elif in_thinking:
            print("</think>\n", end="", flush=True)
            in_thinking = False

        if content_token:
            print(content_token, end="", flush=True)
            reply += content_token

    if in_thinking:
        # Stream ended while still inside a thinking block
        print("</think>\n", end="", flush=True)

    if not reply:
        if thinking:
            print("\n[The model produced only reasoning output with no final answer. "
                  "Try QWEN_REASONING=off to disable thinking mode.]")
        else:
            print("[The model returned an empty response.]")

    print()
    return reply


def main():
    client = create_client()

    print("\n" + "=" * 60)
    print(f"  Qwen Chat  |  model : {MODEL_NAME}")
    print(f"  Server     |  {BASE_URL}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    # Maintain full conversation history for multi-turn chat
    history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            print("\nQwen: ", end="", flush=True)
            reply = chat_once(client, history)
        except Exception as exc:
            print(f"\n[Error] Could not reach the model server: {exc}")
            print("Make sure your inference server is running and OPENAI_API_BASE is set correctly.\n")
            history.pop()   # remove the unanswered user message
            continue

        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
