import json
import time
import uuid
from typing import AsyncIterator, Any


def to_openai_completion(ollama_resp: dict, model: str) -> dict:
    message = ollama_resp.get("message", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            },
            "finish_reason": "stop" if ollama_resp.get("done") else None,
        }],
        "usage": {
            "prompt_tokens": ollama_resp.get("prompt_eval_count", 0) or 0,
            "completion_tokens": ollama_resp.get("eval_count", 0) or 0,
            "total_tokens": (ollama_resp.get("prompt_eval_count", 0) or 0)
                            + (ollama_resp.get("eval_count", 0) or 0),
        },
    }


def to_openai_chunk(ollama_chunk: dict, model: str, first: bool) -> dict:
    content = ollama_chunk.get("message", {}).get("content", "")
    done = ollama_chunk.get("done", False)
    delta: dict[str, str] = {}
    if first:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": "stop" if done else None,
        }],
    }


async def convert_ndjson_to_sse(
    byte_stream: AsyncIterator[bytes], model: str,
) -> AsyncIterator[bytes]:
    buffer = b""
    first = True
    async for chunk in byte_stream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                chunk_data = to_openai_chunk(data, model, first)
                first = False
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode()
            except json.JSONDecodeError:
                continue
    yield b"data: [DONE]\n\n"
