from typing import Protocol


class CatalogServiceInterface(Protocol):
    def get_datasets(
        self,
        include_derived: bool = False,
        org_id: str | None = None,
    ) -> list[dict[str, object]]: ...
    def get_models(self) -> list[dict[str, str]]: ...
    def get_trained_models(self, org_id: str | None = None) -> dict[str, object]: ...
    def upload_user_dataset(
        self,
        file_content: bytes,
        filename: str | None,
        name: str | None = None,
        description: str | None = None,
        org_id: str | None = None,
    ) -> dict[str, object]: ...
    def get_dataset_preview(
        self,
        ref: str,
        limit: int = 25,
        org_id: str | None = None,
    ) -> dict[str, object]: ...
