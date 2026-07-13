# Restore React Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the original React frontend experience and wire it back to the Python backend.

**Architecture:** Restore the tracked `frontend/` Vite React app from `HEAD`, then verify its API and WebSocket expectations against the Python FastAPI backend. Keep the Python-native HTML UI as a fallback only; the React app should be the main frontend again.

**Tech Stack:** React 18, Vite, TypeScript, Zustand, SockJS/STOMP, FastAPI Python backend.

---

### Task 1: Restore Source Files

**Files:**
- Restore: `frontend/**`

- [ ] Restore the deleted tracked React frontend from git.
- [ ] Confirm `frontend/package.json`, `frontend/src/App.tsx`, `frontend/src/api/stompClient.ts`, and `frontend/vite.config.ts` exist.

### Task 2: Check Backend Compatibility

**Files:**
- Read: `frontend/src/api/index.ts`
- Read: `frontend/src/api/stompClient.ts`
- Read: `backend-python/app.py`

- [ ] Compare the frontend REST API paths with Python backend routes.
- [ ] Compare frontend SockJS/STOMP destinations with Python backend WebSocket handlers.
- [ ] Patch only compatibility gaps required for startup and basic chat.

### Task 3: Verify Build

**Files:**
- Read/Modify if needed: `frontend/package.json`
- Read/Modify if needed: `frontend/tsconfig.json`

- [ ] Install or reuse frontend dependencies.
- [ ] Run `npm.cmd run build` in `frontend/`.
- [ ] Fix TypeScript/build errors caused by the restoration or Python backend compatibility.

### Task 4: Runtime Handoff

**Files:**
- Modify if needed: `run.py`
- Modify if needed: `README.md`

- [ ] Decide whether the restored React app runs through Vite dev server or static build serving.
- [ ] Document the correct startup URL and commands.
- [ ] Verify the app loads and can reach backend health/chat endpoints.
