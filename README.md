# Onanana (AI Warp Tool)

Ollama-compatible proxy on `:11435` that routes requests to local Ollama or a cloud API (native `/api/*` + OpenAI-compatible `/v1/*`).

```
Client -> :11435 -> OllamaProvider -> localhost:11434 (no auth)
                                   -> cloud URL (Bearer token)
```

Routing is determined by **model name suffix** (default `-cloud`) or **`?source=` query param**.

## Prerequisites

- **Local** — [Ollama](https://ollama.com) must be installed and running on the target host (default `http://localhost:11434`)
- **Cloud** — one or more API keys saved in `secrets/keys.txt` (one per line, see [`secrets/keys.txt.example`](secrets/keys.txt.example))

## Quick start

```bash
pip install -r requirements/requirements-dev.txt
python -m uvicorn apis.main:app --host 0.0.0.0 --port 11435
```

## How routing works

| Model name | Default route | model sent to backend |
|---|---|---|
| `gemma4:26b` | local (`:11434`) | `gemma4:26b` |
| `gemma4:31b-cloud` | cloud | `gemma4:31b` (suffix stripped) |
| any + `?source=local` | local | original name kept, even with `-cloud` suffix |
| any + `?source=cloud` | cloud | suffix stripped if present |

The cloud suffix is `-cloud` (hardcoded in `src/onanana/providers/ollama.py`).

Cloud auth: token from `secrets/keys.txt` (round-robin with health checks) → `WARP_CLOUD_API_KEY` env var → `503`.

## Key locking & unlocking

When a cloud key gets a `429` (rate limited) or times out 3 times, it is locked into `secrets/ollama_keys_lock.txt` so it won't be reused. Keys are unlocked automatically in **3 ways**:

| # | Trigger | When |
|---|---|---|
| 1 | Background task | Every **10 minutes** (`apis/main.py` cleanup loop) |
| 2 | On every endpoint call | Before each request, lock file is checked and expired entries removed |
| 3 | Lock file expiry | **5 hours** after a key was locked (`LOCK_DURATION` in `keys_manager.py`) |

When all keys are locked, the proxy returns `429` instead of `500`.

## Configuration

`WARP_` env vars or `secrets/.env` file (see `src/onanana/config.py`):

| Variable | Default | Description |
|---|---|---|
| `WARP_HOST` | `0.0.0.0` | Bind address |
| `WARP_PORT` | `11435` | Listen port |
| `WARP_LOCAL_OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama |
| `WARP_CLOUD_OLLAMA_BASE_URL` | `""` | Cloud API endpoint |
| `WARP_CLOUD_API_KEY` | `""` | Fallback Bearer token |
| `WARP_KEYS_FILE_PATH` | `secrets/keys.txt` | API tokens file |
| `WARP_LOCK_FILE_PATH` | `secrets/ollama_keys_lock.txt` | Key lock file (auto-resets every 5h) |
| `WARP_CLOUD_MODEL_SUFFIX` | `-cloud` | Suffix for cloud model routing (defined in config, wired in source) |

## Architecture

```
apis/main.py                  FastAPI app, imports src/ package
src/onanana/
  config.py                   Pydantic settings (load_dotenv secrets/.env)
  keys_manager.py             Loads tokens, round-robin, health checks, key locking
  providers/ollama.py         OllamaProvider — proxy methods with retry & timeout key bans
  ollama/
    models.py                 Pydantic schemas (request/response)
    request.py                Request builder
examples/
  use_pakage.ipynb            Interactive package usage examples
  ai_warp_tool_api.ipynb      Interactive API endpoint examples
  chat_stream.py              Chat via the proxy
  chat_ollama_api_key.py      Direct cloud API call with key
tests/
  check_ai_warp_tool_api.py  23 tests for API endpoints & provider
```

## Examples

```bash
# Chat (streaming) via the proxy
python examples/chat_stream.py

# Direct cloud API call with key
python examples/chat_ollama_api_key.py
```

See [`examples/use_pakage.ipynb`](examples/use_pakage.ipynb) and [`examples/ai_warp_tool_api.ipynb`](examples/ai_warp_tool_api.ipynb) for interactive examples.
