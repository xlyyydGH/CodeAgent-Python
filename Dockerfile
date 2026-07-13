# =============================================================================
# CodeAgent Python
# Runs the Python backend on port 8080 and serves the restored React frontend.
# The existing python-service remains available internally on port 8000.
# =============================================================================

FROM node:26-slim AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
RUN npm run build

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="CodeAgent Python"
LABEL org.opencontainers.image.description="Python-native browser coding agent"
LABEL org.opencontainers.image.source="https://github.com/xlyyydGH/CodeAgent-Python"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_SERVICE_URL=http://127.0.0.1:8000 \
    ZHIKUN_DATA_DIR=/app/backend-python/data \
    LOG_DIR=/app/log

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git ripgrep libmagic1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r zhikun && useradd -r -g zhikun -d /app -s /bin/sh zhikun

WORKDIR /app

COPY backend-python ./backend-python/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist/
COPY python-service/src ./python-service/src/
COPY python-service/requirements.txt python-service/requirements.lock python-service/pyproject.toml ./python-service/
COPY configuration ./configuration/
COPY run.py ./run.py

RUN pip install --no-cache-dir -r backend-python/requirements.txt && \
    pip install --no-cache-dir -r python-service/requirements.txt && \
    mkdir -p /app/backend-python/data /app/workspace /app/log && \
    chown -R zhikun:zhikun /app

USER zhikun

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8080/api/health || exit 1

ENTRYPOINT ["python", "run.py"]
