"""Celery application configuration.

Broker and result backend both use Redis. The REDIS_URL env var
defaults to a local Redis instance for development.

Usage:
    # Start worker
    celery -A backend.celery_app worker --loglevel=info --concurrency=2

    # Start beat (if scheduled tasks are added later)
    celery -A backend.celery_app beat --loglevel=info
"""

import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "jubilee",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.training.tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 hours
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=3900,  # 1h5m hard limit
)
