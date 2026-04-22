"""Pytest conftest — ensure project paths are bootstrapped before any import."""

from __future__ import annotations

from backend.shared.settings import bootstrap_paths

bootstrap_paths()
