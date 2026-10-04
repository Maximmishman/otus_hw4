"""OpenAI-compatible API gateway that exposes the RAG pipeline to OpenWebUI.

Run:  python Scripts/05_api.py
      (or: uvicorn 05_api:app --host 127.0.0.1 --port 8000)

Endpoints:
    GET  /health                -> status
    GET  /v1/models             -> list of available "models" (the RAG bot)
    POST /v1/chat/completions    -> OpenAI Chat Completions (streaming + non-streaming)

OpenWebUI: Settings -> Connections -> OpenAI API
    Base URL: http://127.0.0.1:8000/v1
    API Key:  rag-local   (any value, see API_KEY in config.py)
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from _engine_provider import get_engine  # noqa: E402

app = FastAPI(title="Enterprise Private GPT (RAG gateway)")

MODEL_ID = "corporate-rag"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": config.LLM_MODEL, "embedding": config.EMBEDDING_MODEL}


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "owned_by": "local"},
            {"id": config.LLM_MODEL, "object": "model", "owned_by": "ollama"},
        ],
    }


def _last_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, list):  # OpenAI multimodal format
                content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
):
    if authorization and not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Expected Bearer token")
    payload = await request.json()
    messages = payload.get("messages", [])
    question = _last_user_message(messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    top_k = int(payload.get("top_k", config.TOP_K))
    engine = get_engine()
    result = engine.answer(question, top_k=top_k)

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # include sources in the answer so they are visible in the UI
    answer_text = result["answer"]
    sources = list(dict.fromkeys(result["sources"]))
    if sources:
        answer_text += "\n\n_" + ", ".join(sources) + "_"

    if payload.get("stream"):
        def event_stream():
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            done = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_sources": sources,
        "x_latency_sec": result["latency_sec"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")
