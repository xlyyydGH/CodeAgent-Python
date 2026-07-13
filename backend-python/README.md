# CodeAgent Python Backend

This directory replaces the original Java/Spring Boot backend at port `8080`.

What it provides:

- FastAPI REST endpoints for the restored React UI, Python fallback UI, and API clients.
- A SockJS/STOMP-compatible `/ws` endpoint for realtime clients.
- The restored React build is served from `/` when `frontend/dist/index.html` exists.
- Python-rendered fallback pages remain available under `/ui`.
- Native Python implementations for chat, REST query/SSE query, LLM tool calls, sessions, session compact/export, session snapshots, permissions, commands, auth/admin/remote control, dialogs, file history/rewind, verification/evidence, browser replay, attachments, plugins/skills/hooks, MCP state/capabilities, memory, swarm orchestration, file search/tree, Git log/diff/blame, complexity, code path/diagram, and change-impact APIs.
- Optional Python analysis service support on `8000`; the main API paths now run directly in `backend-python/`.
- Lightweight JSON state persistence in `backend-python/data/state.json`.

Local startup:

```bash
python run.py
```

Configure the iFlytek/XFYun MaaS API in the project `.env`:

```env
LLM_PROVIDER_XFYUN_API_KEY=your-real-api-key
# Optional overrides; defaults match the MaaS card from the screenshot.
# LLM_PROVIDER_XFYUN_BASE_URL=https://maas-api.cn-huabei-1.xf-yun.com/v2
# LLM_PROVIDER_XFYUN_MODEL=xopqwen36v35b
```

Manual backend startup:

```bash
cd backend-python
python -m pip install -r requirements.txt
PYTHON_SERVICE_URL=http://127.0.0.1:8000 python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Current migration status:

- The Java backend is no longer used by the Python runner, `Dockerfile`, or `docker-compose.yml`.
- The original React frontend has been restored in `frontend/` and is served by the Python backend after `npm run build`.
- Chat uses an OpenAI-compatible Python LLM path when credentials are configured, with a no-key local fallback for offline development.
- Read-only workspace tools are exposed to the LLM through OpenAI-compatible tool definitions and executed inside the Python backend.
- Verification uses Python-native per-file checks (`py_compile` for Python, local `npx` TypeScript/ESLint/Vitest when available), computes Java-compatible `signal` values, and stores evidence bundles/blobs.
- Current regression gate: `python-service\venv\Scripts\python.exe -m pytest backend-python\tests -q`.
