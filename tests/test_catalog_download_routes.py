import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.catalog import routes
from fastapi import HTTPException


class _FakeSession:
    def __init__(self, *, model=None, version=None, dataset=None):
        self._model = model
        self._version = version
        self._dataset = dataset
        self._entity = None
        self._filters = {}

    def query(self, entity):
        self._entity = entity
        self._filters = {}
        return self

    def filter_by(self, **kwargs):
        self._filters = kwargs
        return self

    def first(self):
        if self._entity is routes.Model and self._filters.get("name") == getattr(self._model, "name", None):
            return self._model
        if (
            self._entity is routes.ModelVersion
            and self._filters.get("model_id") == getattr(self._model, "id", None)
            and self._filters.get("is_current") is True
        ):
            return self._version
        if self._entity is routes.Dataset and self._filters.get("name") == getattr(self._dataset, "name", None):
            return self._dataset
        return None


class _SessionContext:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_model_redirects_to_presigned_url(monkeypatch):
    model = SimpleNamespace(id="model-1", name="nn_wider_v2")
    version = SimpleNamespace(version=1, storage_key="models/nn_wider_v2/v1/artifact.joblib")
    store = SimpleNamespace(
        get_presigned_url=lambda key: f"https://files.example.test/{key}"
    )

    monkeypatch.setattr(
        routes,
        "get_db_session",
        lambda: _SessionContext(_FakeSession(model=model, version=version)),
    )
    monkeypatch.setattr(routes, "get_artifact_store", lambda: store)

    response = asyncio.run(routes.download_model("nn_wider_v2"))

    assert response.status_code == 307
    assert response.headers["location"] == "https://files.example.test/models/nn_wider_v2/v1/artifact.joblib"


def test_download_dataset_redirects_to_presigned_url(monkeypatch):
    dataset = SimpleNamespace(
        name="predictions_latest",
        properties={"storage_key": "datasets/predictions_latest/data.parquet"},
    )
    store = SimpleNamespace(
        get_presigned_url=lambda key: f"https://files.example.test/{key}"
    )

    monkeypatch.setattr(
        routes,
        "get_db_session",
        lambda: _SessionContext(_FakeSession(dataset=dataset)),
    )
    monkeypatch.setattr(routes, "get_artifact_store", lambda: store)

    response = asyncio.run(routes.download_dataset("predictions_latest"))

    assert response.status_code == 307
    assert response.headers["location"] == "https://files.example.test/datasets/predictions_latest/data.parquet"


def test_download_model_raises_when_missing():
    with pytest.raises(HTTPException, match="Model not found"):
        asyncio.run(routes.download_model("missing-model"))
