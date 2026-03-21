"""
S3-compatible artifact store for model weights.

Uses Cloudflare R2 when configured, falls back to local filesystem for
development. All callers go through `get_artifact_store()` which returns
a singleton configured from Settings.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from backend.shared.settings import get_settings


class ArtifactStoreProtocol(Protocol):
    def upload(self, local_path: Path, key: str) -> str: ...
    def download(self, key: str, local_path: Path) -> Path: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def checksum(self, local_path: Path) -> str: ...


class R2ArtifactStore:
    """Cloudflare R2 backend via boto3 S3 client."""

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
    ):
        self._bucket = bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                signature_version="s3v4",
            ),
            region_name="auto",
        )

    def upload(self, local_path: Path, key: str) -> str:
        self._client.upload_file(str(local_path), self._bucket, key)
        return key

    def download(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, str(local_path))
        return local_path

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def checksum(self, local_path: Path) -> str:
        return _sha256(local_path)


class LocalArtifactStore:
    """Local filesystem fallback for development without R2 credentials."""

    def __init__(self, base_dir: Path):
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        return self._base / key

    def upload(self, local_path: Path, key: str) -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != dest.resolve():
            shutil.copy2(local_path, dest)
        return key

    def download(self, key: str, local_path: Path) -> Path:
        src = self._resolve(key)
        if not src.exists():
            raise FileNotFoundError(f"Local artifact not found: {src}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != local_path.resolve():
            shutil.copy2(src, local_path)
        return local_path

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def checksum(self, local_path: Path) -> str:
        return _sha256(local_path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def get_artifact_store() -> R2ArtifactStore | LocalArtifactStore:
    settings = get_settings()
    if settings.r2_enabled:
        return R2ArtifactStore(
            endpoint_url=settings.R2_ENDPOINT_URL,
            access_key_id=settings.R2_ACCESS_KEY_ID,
            secret_access_key=settings.R2_SECRET_ACCESS_KEY.get_secret_value(),
            bucket_name=settings.R2_BUCKET_NAME,
        )
    return LocalArtifactStore(settings.trained_models_dir)
