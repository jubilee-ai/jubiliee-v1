"""Pytest conftest — ensure project paths are bootstrapped before any import."""

from __future__ import annotations

import os

# Default tests to the sklearn training toolkit unless the runner sets this explicitly.
os.environ.setdefault("TRAINING_USE_H2O_ONLY", "false")

from backend.shared.settings import bootstrap_paths

bootstrap_paths()
