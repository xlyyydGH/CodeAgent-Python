# Contributing to CodeAgent Python

## Development Setup

Prerequisites:

- Python 3.11 or newer
- Docker, optional for container runs
- Git

Install dependencies:

```bash
python -m pip install -r backend-python/requirements.txt
python -m pip install -r python-service/requirements.txt
```

Start the local Python services:

```bash
python run.py
```

Project structure:

| Directory | Description |
| --- | --- |
| `backend-python/` | FastAPI backend, Python UI, API tests |
| `python-service/` | Python analysis service |
| `configuration/` | Shared runtime configuration |
| `docs/` | Documentation and migration notes |
| `scripts/` | Utility scripts |

## Test Gate

Run this before submitting changes:

```powershell
python-service\venv\Scripts\python.exe -m pytest backend-python\tests -q
```

Also compile edited Python modules when changing backend code:

```powershell
python-service\venv\Scripts\python.exe -m py_compile backend-python\app.py
```

## Code Style

- Use Python type hints for new public helpers.
- Keep handlers small and reuse local helpers where possible.
- Preserve API compatibility unless a migration note explains the change.
- Add or update regression tests for every endpoint or user-facing behavior change.

## Pull Requests

Keep PRs focused and include:

- What changed
- Why it changed
- Test results

By submitting a pull request, you agree that your contributions will be licensed under the MIT License.
