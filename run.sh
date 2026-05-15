#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source ".venv313/bin/activate"
exec uvicorn main:app --reload --host 127.0.0.1 --port 8000 --app-dir "$ROOT/backend"
