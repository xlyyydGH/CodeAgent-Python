from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"


def env_with(**updates: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(updates)
    return env


def start_process(name: str, args: list[str], cwd: Path, log_file: Path, env: dict[str, str]) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("ab")
    print(f"[INFO] Starting {name}: {' '.join(args)}")
    return subprocess.Popen(args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)


def terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 8
    for process in processes:
        remaining = max(0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    backend_port = os.getenv("BACKEND_PORT", "8080")
    python_port = os.getenv("PYTHON_PORT", "8000")
    reload_flag = os.getenv("ZHIKUN_RELOAD", "false").lower() == "true"
    LOG_DIR.mkdir(exist_ok=True)

    python_service_args = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host",
        os.getenv("PYTHON_SERVICE_HOST", "127.0.0.1"),
        "--port",
        python_port,
    ]
    backend_args = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        os.getenv("BACKEND_HOST", "0.0.0.0"),
        "--port",
        backend_port,
    ]
    if reload_flag:
        python_service_args.append("--reload")
        backend_args.append("--reload")

    processes = [
        start_process(
            "python-service",
            python_service_args,
            ROOT / "python-service",
            LOG_DIR / "python-service-console.log",
            env_with(PYTHONPATH=str(ROOT / "python-service" / "src")),
        ),
        start_process(
            "backend-python",
            backend_args,
            ROOT / "backend-python",
            LOG_DIR / "backend-python-console.log",
            env_with(
                PYTHONPATH=str(ROOT / "backend-python"),
                ZHIKUN_DATA_DIR=os.getenv("ZHIKUN_DATA_DIR", str(ROOT / "backend-python" / "data")),
                PYTHON_SERVICE_URL=os.getenv("PYTHON_SERVICE_URL", f"http://127.0.0.1:{python_port}"),
            ),
        ),
    ]

    def handle_signal(_signum: int, _frame: object) -> None:
        terminate(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print()
    print("CodeAgent Python is running")
    print(f"  UI/API          http://localhost:{backend_port}")
    print(f"  Python service  http://localhost:{python_port}")
    print(f"  Logs            {LOG_DIR}")
    print()

    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    terminate(processes)
                    return code
            time.sleep(1)
    finally:
        terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
