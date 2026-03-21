#!/bin/sh
set -e

if [ -n "$START_COMMAND" ]; then
    exec $START_COMMAND
fi

python -m alembic upgrade head
exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
