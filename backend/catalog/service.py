import uuid
from typing import Optional

from backend.catalog import repository


def get_datasets(
    include_derived: bool = False,
    user_id: Optional[uuid.UUID] = None,
    org_id: Optional[uuid.UUID] = None,
) -> list[dict[str, object]]:
    return repository.get_datasets(
        include_derived=include_derived,
        user_id=user_id,
        org_id=org_id,
    )


def get_models() -> list[dict[str, str]]:
    return repository.get_models()


def get_trained_models(
    user_id: Optional[uuid.UUID] = None,
    org_id: Optional[uuid.UUID] = None,
) -> dict[str, object]:
    return repository.get_trained_models(user_id=user_id, org_id=org_id)
