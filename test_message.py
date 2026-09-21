from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
stream = c.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
    max_tokens=512,
    stream=True,
)
content, reasoning = "", ""
for chunk in stream:
    print("CHUNK:", chunk.model_dump())
    d = chunk.choices[0].delta
    if getattr(d, "content", None):
        content += d.content
    if getattr(d, "reasoning_content", None):
        reasoning += d.reasoning_content
print("REASONING:", reasoning)
print("REPLY:", content)
