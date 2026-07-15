import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from src.onanana.config import settings
from src.onanana.keys_manager import KeysManager
from src.onanana.providers.ollama import OllamaProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CLEANUP_INTERVAL = 600  # 10 minutes

km = KeysManager(settings.keys_file_path, cloud_base_url=settings.cloud_ollama_base_url,
                 lock_path=settings.lock_file_path)
km.load_keys()
client = httpx.AsyncClient(timeout=300.0)
provider = OllamaProvider(
    local_base_url=settings.local_ollama_base_url,
    cloud_base_url=settings.cloud_ollama_base_url,
    keys_manager=km,
    client=client,
    cloud_api_key=settings.cloud_api_key,
    lock_path=settings.lock_file_path,
)


async def cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        removed = km.cleanup_expired_locks()
        if removed:
            logger.info("Auto-cleanup removed %d expired lock(s)", removed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()
    await client.aclose()
    await km.close()


app = FastAPI(title="AI Warp Tool", lifespan=lifespan)


@app.exception_handler(RuntimeError)
async def no_key_handler(request: Request, exc: RuntimeError):
    if "No API key available" in str(exc):
        return JSONResponse(status_code=429, content={"error": "No API keys available - all keys locked or missing"})
    return JSONResponse(status_code=500, content={"error": str(exc)})


def _resolve_v1_path(rest: str, source: str | None) -> tuple[str, str | None]:
    """For GET /v1/models/{name}, infer cloud + strip -cloud from path when needed."""
    if not rest.startswith("models/"):
        return rest, source
    model = rest[len("models/"):]
    if source is None and OllamaProvider.is_cloud_model(model):
        source = "cloud"
    if source == "cloud":
        return f"models/{OllamaProvider.strip_cloud_suffix(model)}", source
    return rest, source


def _stream_media_type(path: str) -> str:
    if path.startswith("v1/"):
        return "text/event-stream"
    return "application/x-ndjson"


async def _forward(
    request: Request,
    path: str,
    *,
    source: str | None = None,
):
    km.cleanup_expired_locks()

    try:
        body = await request.json()
    except Exception:
        body = {}

    if "messages" in body and "generate" in path:
        for msg in body["messages"]:
            if msg.get("role") == "system":
                body["system"] = msg["content"]
            elif msg.get("role") == "user":
                body["prompt"] = msg["content"]
        body.pop("messages", None)
        body.pop("message", None)

    method = request.method
    is_stream = body.get("stream", False) if method == "POST" else False

    if method in ("GET", "HEAD"):
        resp = await provider.proxy_get(path, source=source or "local", method=method)
    elif method == "DELETE":
        resp = await provider.proxy_delete(path, body, source=source)
    else:
        resp = await provider.proxy_request(path, body, stream=is_stream, source=source)

    if method == "HEAD":
        await resp.aread()
        return Response(status_code=resp.status_code)

    if is_stream:
        return StreamingResponse(resp.aiter_bytes(), media_type=_stream_media_type(path))
    await resp.aread()
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/api/version")
async def version(source: str = Query("local", pattern="^(local|cloud)$")):
    km.cleanup_expired_locks()
    resp = await provider.proxy_get("api/version", source=source)
    await resp.aread()
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/api/tags")
async def tags(source: str = Query("local", pattern="^(local|cloud)$")):
    km.cleanup_expired_locks()
    resp = await provider.proxy_get("api/tags", source=source)
    await resp.aread()
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "DELETE", "HEAD"])
async def proxy_api(
    request: Request,
    rest: str,
    source: str = Query(None, pattern="^(local|cloud)$"),
):
    return await _forward(request, f"api/{rest}", source=source)


@app.api_route("/v1/{rest:path}", methods=["GET", "POST", "DELETE", "HEAD"])
async def proxy_v1(
    request: Request,
    rest: str,
    source: str = Query(None, pattern="^(local|cloud)$"),
):
    """OpenAI-compatible endpoints: chat/completions, completions, embeddings, models, responses, etc."""
    rest, resolved = _resolve_v1_path(rest, source)
    return await _forward(request, f"v1/{rest}", source=resolved)


if __name__ == "__main__":
    uvicorn.run("apis.main:app", host=settings.warp_host, port=settings.warp_port)
