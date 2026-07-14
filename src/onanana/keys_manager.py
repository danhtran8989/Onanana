import asyncio
import datetime
import logging
import random
from pathlib import Path
import sys
import httpx

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

LOCK_SEPARATOR = "||"
LOCK_DURATION = datetime.timedelta(hours=5)

class KeysManager:
    def __init__(self, file_path: str, cloud_base_url: str = "",
                 lock_path: str = ""):
        self._file_path = Path(file_path)
        self._lock_path = Path(lock_path) if lock_path else None
        self._cloud_base = cloud_base_url.rstrip("/")
        self._keys: list[str] = []
        self._locked_keys: set[str] = set()
        self._index: int = 0
        self._lock = asyncio.Lock()
        # Increased timeout slightly to accommodate chat API response time
        self._client = httpx.AsyncClient(timeout=15.0)
        self._healthy_keys: list[str] = []

    def _load_locked_keys(self) -> set[str]:
        locked: set[str] = set()
        if not self._lock_path or not self._lock_path.exists():
            return locked
        now = datetime.datetime.now(datetime.timezone.utc)
        valid_lines: list[str] = []
        expired_count = 0
        for line in self._lock_path.read_text().splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                valid_lines.append(raw)
                continue
            if LOCK_SEPARATOR in raw:
                key, ts_str = raw.split(LOCK_SEPARATOR, 1)
                try:
                    ts = datetime.datetime.fromisoformat(ts_str)
                    if now - ts > LOCK_DURATION:
                        expired_count += 1
                        continue
                    locked.add(key)
                    valid_lines.append(raw)
                    continue
                except ValueError:
                    pass
            locked.add(raw)
            valid_lines.append(raw)
        if expired_count:
            content = "\n".join(valid_lines)
            if content:
                content += "\n"
            self._lock_path.write_text(content)
            logger.info("Cleaned %d expired key(s) from %s", expired_count, self._lock_path)
        if locked:
            logger.info("Loaded %d locked key(s) from %s", len(locked), self._lock_path)
        return locked

    def load_keys(self) -> list[str]:
        if not self._file_path.exists():
            logger.warning("Keys file not found: %s", self._file_path)
            return []
        keys = [
            line.strip()
            for line in self._file_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self._keys = list(keys)
        self._locked_keys = self._load_locked_keys()
        self._healthy_keys = [k for k in self._keys if k not in self._locked_keys]
        if self._locked_keys:
            logger.info("Filtered out %d locked key(s), %d healthy",
                        len(self._locked_keys), len(self._healthy_keys))
        logger.info("Loaded %d key(s) from %s", len(keys), self._file_path)
        return keys

    async def check_key_health(self, key: str) -> bool:
        if not self._cloud_base:
            return True
        try:
            url = f"{self._cloud_base}/api/chat"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            }
            # Query request matching the Ollama chat API structure
            payload = {
                "model": "gemma3:4b-cloud",
                "messages": [
                    {
                        "role": "user",
                        "content": "test"
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }
            r = await self._client.post(url, headers=headers, json=payload)
            # Consider healthy only if response code is exactly 200
            return r.status_code == 200
        except Exception as e:
            logger.debug("Health check failed for key: %s", e)
            return False

    async def refresh_healthy_keys(self) -> list[str]:
        self._locked_keys = self._load_locked_keys()
        healthy = []
        for key in self._keys:
            if key in self._locked_keys:
                continue
            if await self.check_key_health(key):
                healthy.append(key)
        async with self._lock:
            self._healthy_keys = healthy
            if self._index >= len(self._healthy_keys):
                self._index = 0
        logger.info("Healthy keys: %d / %d", len(healthy), len(self._keys))
        return healthy

    async def get_next_healthy_key(self) -> str | None:
        self.cleanup_expired_locks()
        async with self._lock:
            if self._healthy_keys:
                key = self._healthy_keys[self._index % len(self._healthy_keys)]
                self._index = (self._index + 1) % len(self._healthy_keys)
                return key

        await self.refresh_healthy_keys()

        async with self._lock:
            if self._healthy_keys:
                key = self._healthy_keys[self._index % len(self._healthy_keys)]
                self._index = (self._index + 1) % len(self._healthy_keys)
                return key
            return None

    def cleanup_expired_locks(self) -> int:
        old_count = len(self._locked_keys)
        self._locked_keys = self._load_locked_keys()
        removed = old_count - len(self._locked_keys)
        if removed:
            healthy_set = set(self._healthy_keys)
            for k in self._keys:
                if k not in self._locked_keys and k not in healthy_set:
                    self._healthy_keys.append(k)
                    healthy_set.add(k)
            self._healthy_keys = [k for k in self._healthy_keys if k not in self._locked_keys]
            logger.info("Unlocked %d expired key(s), %d still locked, %d healthy",
                        removed, len(self._locked_keys), len(self._healthy_keys))
        return removed

    async def close(self):
        await self._client.aclose()


if __name__ == "__main__":
    # Configure logging for the main execution block
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def main():
        # Initialize with the paths and URLs from your configuration
        manager = KeysManager(
            file_path="secrets/keys.txt",
            cloud_base_url="https://api.ollama.com"  # Change to your actual cloud endpoint if different
        )
        
        keys = manager.load_keys()
        if not keys:
            logger.warning("No keys loaded. Exiting.")
            await manager.close()
            return
            
        logger.info("Checking health of keys via Ollama chat API (gemma3:4b-cloud)...")
        healthy_keys = await manager.refresh_healthy_keys()
        
        if healthy_keys:
            # Randomly release/select a healthy key
            selected_key = random.choice(healthy_keys)
            logger.info(f"Successfully found {len(healthy_keys)} healthy key(s).")
            logger.info(f"Randomly selected healthy key: {selected_key}")
        else:
            logger.warning("No healthy keys found. Please check your keys.txt or cloud endpoint.")
            
        await manager.close()

    asyncio.run(main())