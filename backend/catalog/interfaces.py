import uuid
from typing import Optional, Protocol


class CatalogServiceInterface(Protocol):
    def get_datasets(
        self,
        include_derived: bool = False,
        user_id: Optional[uuid.UUID] = None,
        org_id: Optional[uuid.UUID] = None,
    ) -> list[dict[str, object]]: ...
    def get_models(self) -> list[dict[str, str]]: ...
    def get_trained_models(
        self,
        user_id: Optional[uuid.UUID] = None,
        org_id: Optional[uuid.UUID] = None,
    ) -> dict[str, object]: ...
