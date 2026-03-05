from backend.catalog import repository


def get_datasets() -> list[dict[str, object]]:
    return repository.get_datasets()


def get_models() -> list[dict[str, str]]:
    return repository.get_models()


def get_trained_models() -> dict[str, object]:
    return repository.get_trained_models()
