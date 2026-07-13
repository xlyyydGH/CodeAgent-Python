# Python Rewrite Plan

This project is being migrated to a Python-only implementation with feature parity as the target.

## Current State

- `backend-python/` is the active runtime backend on port `8080`.
- `python-service/` remains the Python analysis service on port `8000`.
- The original `backend/` Java/Spring source tree has been removed from the Python edition.
- The original `frontend/` React/TypeScript source tree has been removed from the Python edition.
- `run.py` and the Docker runtime now run Python-only services.

## Non-Negotiable Migration Rule

A module is not considered migrated just because an endpoint exists. It is migrated only when:

1. The Python implementation preserves the user-facing behavior.
2. The public API or UI route has regression tests.
3. Runtime state is persisted or recovered in the same situations as the original feature.
4. Removing legacy source must not break the Python app.

## Backend Migration Tracks

| Track | Former Java Reference | Python Target | Status |
| --- | --- | --- | --- |
| HTTP API surface | `controller`, `websocket` | `backend-python/app.py` | Implemented, expanding parity tests |
| Session state | `session`, `state`, `service/*Repository` | `backend-python` state store | Implemented for create/list/detail/delete/resume/compact/export/snapshots |
| LLM providers | `llm`, `engine/Query*` | Python OpenAI-compatible client | Implemented for UI, REST, SSE, and WebSocket entrypoints |
| Tools | `tool`, `permission`, `security`, `sandbox` | Python tool registry and permission pipeline | Implemented for read-only workspace tools and permission rules |
| Commands | `command` | Python slash command router | Implemented |
| File/Git APIs | `FileController`, `GitService` | Python file tree/search and Git subprocess adapters | Implemented |
| Code analysis APIs | Python proxy + Java callers | Python complexity, code path/diagram, and change-impact endpoints | Implemented |
| MCP | `mcp` | Python MCP client/server adapters | Implemented for stateful servers/resources/prompts/capabilities/local invocation |
| Plugins/skills | `plugin`, `skill`, `hook` | Python plugin, skill, and hook loader | Implemented |
| Memory | `memdir`, `prompt`, `context` | Python memory and prompt services | Implemented |
| Verification | `verify`, browser replay, evidence | Python verification services | Implemented for per-file checks, Signal, evidence, browser replay |
| Coordinator/swarm | `coordinator`, agent tools | Python worker orchestration | Implemented lightweight stateful orchestration |
| Config/auth/admin | `config`, `bridge`, `keybinding` | Python config and local auth services | Implemented for config/auth/admin/remote control |

## Frontend Migration Tracks

| Former React/TypeScript Reference | Python Target | Status |
| --- | --- | --- |
| Chat layout and message list | Server-rendered Python UI | Implemented |
| Session/sidebar/task panels | Server-rendered Python UI | Basic sessions/activity pages implemented |
| Config/model/permission panels | Server-rendered Python UI | Basic settings page implemented |
| File tree and autocomplete | Server-rendered Python UI + API endpoints | Basic file search page implemented |
| Visualizations | Python-generated Mermaid/HTML views | API implemented, richer Python UI pending |
| APOS/activity/evidence panels | Python views | Basic activity/verification pages implemented |
| MCP/plugin/memory panels | Python views | Basic MCP/memory pages implemented |

## Verification Gate

The Python edition is validated with:

```powershell
python-service\venv\Scripts\python.exe -m pytest backend-python\tests -q
```

## Completed in This Migration Pass

- Added a Python-rendered chat UI at `/`.
- Added Python-rendered replacement pages for sessions, settings, file search, activity, verification/evidence, MCP, and memory.
- Added `/ui/chat` form handling without JavaScript or multipart dependencies.
- Replaced the WebSocket placeholder chat response with a Python OpenAI-compatible LLM path.
- Added OpenAI-compatible LLM tool definitions and a Python tool-call execution loop.
- Added Python read-only tools for listing, reading, and searching workspace files.
- Added Python slash command routing for help/status/model/compact/files/read/search.
- Added Python-native `/api/files/tree`, `/api/git/log`, `/api/git/diff`, `/api/git/blame`, `/api/code-quality/complexity`, and `/api/analysis/change-impact`.
- Made WebSocket control messages stateful for model, permission mode, permission replies, MCP operations, rewind, elicitation, and activity save/update.
- Added persisted file history snapshots, unified diffs, and real rewind writes through Python endpoints and WebSocket.
- Added Python EvidenceBundle/blob storage for `/api/verify/run-checks` and `/api/evidence/*`.
- Replaced verification acceptance stubs with Python-native per-file checks, heuristic impact data, Java-compatible Signal calculation, and legacy-check response mapping.
- Added stateful browser replay timeline storage for `/api/browser/replay/{sessionId}`.
- Added Python session snapshot save/resume/delete endpoints.
- Added Java-compatible session compact/export behavior including downloadable JSON/Markdown exports.
- Added filesystem-backed skill discovery plus state/filesystem plugin discovery.
- Added Java-compatible memory `entries`, `/api/memory/all`, and delete/update flows.
- Added stateful MCP servers/resources/prompts/capabilities, local tool invocation, hooks, dialog decisions, swarm worker orchestration, auth/admin/remote control, multipart attachments, and REST/SSE query persistence.
- Made local and Docker startup default to Python-only runtime services.
- Replaced local start/stop shell scripts with the Python `run.py` runner and removed React dist serving from the Python backend.
- Removed the legacy `backend/`, `frontend/`, and `workspace/app/frontend` source trees from this Python edition.
- Kept no-key local fallback behavior so the app remains runnable without external credentials.

## Latest Verification

```powershell
python-service\venv\Scripts\python.exe -m pytest backend-python\tests -q
# 38 passed
```
