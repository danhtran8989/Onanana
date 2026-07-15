from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.onanana.ollama.models import (
    ChatRequest,
    CopyRequest,
    CreateRequest,
    DeleteRequest,
    EmbedRequest,
    GenerateRequest,
    PullRequest,
    PushRequest,
    ShowRequest,
)

logger = logging.getLogger(__name__)

ENDPOINTS = {
    "generate": {"method": "POST", "model": GenerateRequest},
    "chat": {"method": "POST", "model": ChatRequest},
    "embeddings": {"method": "POST", "model": EmbedRequest},
    "embed": {"method": "POST", "model": EmbedRequest},
    "create": {"method": "POST", "model": CreateRequest},
    "pull": {"method": "POST", "model": PullRequest},
    "push": {"method": "POST", "model": PushRequest},
    "show": {"method": "POST", "model": ShowRequest},
    "copy": {"method": "POST", "model": CopyRequest},
    "delete": {"method": "DELETE", "model": DeleteRequest},
}

# OpenAI-compatible paths (method inference for request builder)
V1_ENDPOINTS = {
    "chat/completions": "POST",
    "completions": "POST",
    "embeddings": "POST",
    "responses": "POST",
    "images/generations": "POST",
    "models": "GET",
}


class OllamaRequestBuilder:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    def build_request(
        self,
        path: str,
        base_url: str,
        body: dict[str, Any] | None,
        *,
        model_override: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Request:
        api_path = path.lstrip("/")
        url = f"{base_url.rstrip('/')}/{api_path}"
        payload = dict(body) if body else {}

        if model_override is not None:
            payload["model"] = model_override

        logger.debug("Building %s %s", self._method_for_path(api_path), url)
        method = self._method_for_path(api_path)
        return self._client.build_request(method, url, json=payload, headers=headers)

    async def send_request(
        self,
        path: str,
        base_url: str,
        body: dict[str, Any] | None,
        *,
        model_override: str | None = None,
        headers: dict[str, str] | None = None,
        stream: bool = True,
    ) -> httpx.Response:
        req = self.build_request(
            path, base_url, body, model_override=model_override, headers=headers
        )
        return await self._client.send(req, stream=stream)

    @staticmethod
    def _method_for_path(api_path: str) -> str:
        path = api_path.lstrip("/")
        for name, info in ENDPOINTS.items():
            if path.endswith(f"api/{name}") or path == f"api/{name}":
                return info["method"]
        for name, method in V1_ENDPOINTS.items():
            if path == f"v1/{name}" or path.endswith(f"/v1/{name}"):
                return method
            # GET /v1/models/{model}
            if name == "models" and (
                path.startswith("v1/models/") or "/v1/models/" in path
            ):
                return "GET"
        return "POST"

    @staticmethod
    def parse_model_field(body: dict[str, Any] | None) -> str:
        return (body or {}).get("model", "")

    @staticmethod
    def is_streaming(body: dict[str, Any] | None) -> bool:
        return bool((body or {}).get("stream", False))

    @staticmethod
    def serialize_stream_chunk(data: dict[str, Any]) -> bytes:
        return (json.dumps(data, ensure_ascii=False) + "\n").encode()
