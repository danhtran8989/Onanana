import datetime
import logging
from pathlib import Path
from typing import Any

import httpx

from src.onanana.keys_manager import KeysManager, LOCK_SEPARATOR
from src.onanana.ollama.request import OllamaRequestBuilder

logger = logging.getLogger(__name__)

CLOUD_SUFFIX = "-cloud"
MAX_RETRIES = 3


class OllamaProvider:
    def __init__(
        self,
        local_base_url: str,
        cloud_base_url: str,
        keys_manager: KeysManager,
        client: httpx.AsyncClient | None = None,
        cloud_api_key: str = "",
        lock_path: str = "",
    ):
        self._local_base = local_base_url.rstrip("/")
        self._cloud_base = cloud_base_url.rstrip("/")
        self._keys_manager = keys_manager
        self._client = client or httpx.AsyncClient(timeout=300.0)
        self._req_builder = OllamaRequestBuilder(self._client)
        self._fallback_api_key = cloud_api_key
        self._lock_path = Path(lock_path) if lock_path else None
        self._timeout_count: dict[str, int] = {}

    def _append_to_lock(self, path: Path | None, key: str):
        if not path:
            return
        existing = path.read_text().splitlines() if path.exists() else []
        already_locked = False
        for line in existing:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if LOCK_SEPARATOR in raw:
                k, _ = raw.split(LOCK_SEPARATOR, 1)
                if k == key:
                    already_locked = True
                    break
            elif raw == key:
                already_locked = True
                break
        if not already_locked:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(path, "a") as f:
                f.write(f"{key}{LOCK_SEPARATOR}{now}\n")
            logger.warning("Key locked -> %s", path.name)

    async def _send_with_retry(
        self, path: str, base: str, body: dict[str, Any] | None, *,
        model_override: str | None = None, headers: dict[str, str] | None = None,
        stream: bool = True, token: str = "",
    ) -> httpx.Response:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await self._req_builder.send_request(
                    path, base, body,
                    model_override=model_override, headers=headers, stream=stream,
                )
            except httpx.TimeoutException:
                logger.warning("Timeout attempt %d/%d for key %s...", attempt, MAX_RETRIES, token[:16])
                if token:
                    self._timeout_count[token] = self._timeout_count.get(token, 0) + 1
                    if self._timeout_count[token] >= MAX_RETRIES:
                        self._append_to_lock(self._lock_path, token)
                        break
        raise httpx.TimeoutException(f"Request failed after {MAX_RETRIES} timeouts")

    @staticmethod
    def is_cloud_model(model: str) -> bool:
        return model.endswith(CLOUD_SUFFIX)

    @staticmethod
    def strip_cloud_suffix(model: str) -> str:
        if model.endswith(CLOUD_SUFFIX):
            return model[: -len(CLOUD_SUFFIX)]
        return model

    async def _resolve_cloud_auth(self) -> tuple[str, str, dict[str, str] | None]:
        if not self._cloud_base:
            raise RuntimeError("Cloud base URL not configured. Set WARP_CLOUD_OLLAMA_BASE_URL.")
        token = await self._keys_manager.get_next_healthy_key()
        if token is None:
            token = self._fallback_api_key
        if token:
            return self._cloud_base, token, {"Authorization": f"Bearer {token}"}
        raise RuntimeError("No API key available for cloud endpoint. Add keys to secrets/keys.txt or set WARP_CLOUD_API_KEY.")

    async def _get_backend_url(self, model: str) -> tuple[str, str]:
        stripped = self.strip_cloud_suffix(model)
        if self.is_cloud_model(model):
            base, _, _ = await self._resolve_cloud_auth()
            return base, stripped
        return self._local_base, stripped

    async def proxy_request(
        self, path: str, json_body: dict[str, Any] | None, stream: bool = True, *, source: str | None = None
    ) -> httpx.Response:
        model = self._req_builder.parse_model_field(json_body)
        stripped_model = self.strip_cloud_suffix(model)
        use_cloud = source == "cloud" if source else self.is_cloud_model(model)

        if use_cloud:
            base, token, headers = await self._resolve_cloud_auth()
            if not token:
                raise RuntimeError("No API key available for cloud endpoint")
            resp = await self._send_with_retry(
                path, base, json_body,
                model_override=stripped_model, headers=headers,
                stream=stream, token=token,
            )
            if resp.status_code == 429 and token:
                self._append_to_lock(self._lock_path, token)
            return resp

        return await self._req_builder.send_request(
            path,
            self._local_base,
            json_body,
            model_override=None,
            stream=stream,
        )

    async def resolve_backend(self, model: str) -> tuple[str, str]:
        return await self._get_backend_url(model)

    async def proxy_get(self, path: str, *, source: str = "local") -> httpx.Response:
        if source == "cloud":
            base, token, headers = await self._resolve_cloud_auth()
            if not token:
                raise RuntimeError("No API key available for cloud endpoint")
            url = f"{base}/{path.lstrip('/')}"
            logger.debug("Proxying cloud GET %s -> %s", path, url)
            resp = await self._send_with_retry(
                path, base, None,
                headers=headers, stream=False, token=token,
            )
            if resp.status_code == 429 and token:
                self._append_to_lock(self._lock_path, token)
            return resp
        url = f"{self._local_base}/{path.lstrip('/')}"
        logger.debug("Proxying local GET %s -> %s", path, url)
        return await self._client.get(url)

    async def proxy_delete(
        self, path: str, json_body: dict[str, Any] | None, *, source: str | None = None
    ) -> httpx.Response:
        model = self._req_builder.parse_model_field(json_body)
        stripped_model = self.strip_cloud_suffix(model)
        use_cloud = source == "cloud" if source else self.is_cloud_model(model)

        if use_cloud:
            base, token, headers = await self._resolve_cloud_auth()
            if not token:
                raise RuntimeError("No API key available for cloud endpoint")
            resp = await self._send_with_retry(
                path, base, json_body,
                model_override=stripped_model, headers=headers,
                stream=False, token=token,
            )
            if resp.status_code == 429 and token:
                self._append_to_lock(self._lock_path, token)
            return resp

        return await self._req_builder.send_request(
            path,
            self._local_base,
            json_body,
            model_override=None,
            stream=False,
        )

    async def close(self):
        await self._client.aclose()
