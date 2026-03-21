"""
One-time backfill script: migrate existing registry.json + catalog.json data
into Postgres tables, and upload existing .joblib files to R2.

Usage:
    cd /path/to/jubiliee-v1
    python scripts/backfill_registry.py

Idempotent: re-running skips rows that already exist.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.shared.settings import get_settings, bootstrap_paths

bootstrap_paths()

from backend.shared.artifact_store import get_artifact_store
from backend.shared.database import get_db_session
from backend.shared.models import Dataset, Model, ModelVersion


def backfill_catalog():
    """Migrate datasets/catalog.json entries into the datasets table."""
    settings = get_settings()
    catalog_path = settings.datasets_catalog_path
    if not catalog_path.exists():
        print("  No catalog.json found, skipping.")
        return

    with open(catalog_path) as f:
        catalog = json.load(f)

    entries = catalog.get("datasets", [])
    count = 0
    with get_db_session() as session:
        for ds in entries:
            name = ds.get("name", ds.get("file", "unknown"))
            existing = session.query(Dataset).filter(Dataset.name == name).first()
            if existing:
                continue

            fmt = ds.get("format", "").upper()
            if fmt in ("CSV", "PARQUET"):
                source_type = "local"
            elif "huggingface" in fmt.lower() or "python" in fmt.lower():
                source_type = "huggingface"
            elif fmt == "SQL":
                source_type = "sql"
            else:
                source_type = "local"

            props = {
                "description": ds.get("description", ""),
                "format": ds.get("format", ""),
                "row_count": ds.get("rows"),
                "columns": ds.get("columns", []),
                "use_case": ds.get("use_case", ""),
                "source_uri": ds.get("file", ""),
                "source": ds.get("source", ""),
            }

            session.add(Dataset(name=name, source_type=source_type, properties=props))
            count += 1

    print(f"  Backfilled {count} datasets from catalog.json.")


def backfill_models():
    """Migrate trained_models/registry.json into models + model_versions tables,
    and upload .joblib files to R2."""
    settings = get_settings()
    registry_path = settings.models_registry_path
    if not registry_path.exists():
        print("  No registry.json found, skipping.")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    models_data = registry.get("models", {})
    store = get_artifact_store()
    count = 0

    with get_db_session() as session:
        for model_name, entry in models_data.items():
            existing = session.query(Model).filter(Model.name == model_name).first()
            if existing:
                print(f"  Model '{model_name}' already exists, skipping.")
                continue

            model = Model(
                name=model_name,
                properties={
                    "model_type": entry.get("model_type", ""),
                    "description": entry.get("description", ""),
                    "feature_names": entry.get("feature_names", []),
                    "target_column": entry.get("target_column", ""),
                },
            )
            session.add(model)
            session.flush()

            # Upload to R2
            storage_key = None
            local_path = entry.get("model_path", "")
            if local_path and Path(local_path).exists():
                safe_name = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in model_name
                )
                storage_key = f"models/{safe_name}/v1/artifact.joblib"
                try:
                    store.upload(Path(local_path), storage_key)
                    print(f"  Uploaded {model_name} -> {storage_key}")
                except Exception as e:
                    print(f"  Warning: Failed to upload {model_name}: {e}")
                    storage_key = None
            else:
                # Try standard path in trained_models/
                safe_name = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in model_name
                )
                alt_path = settings.trained_models_dir / f"{safe_name}.joblib"
                if alt_path.exists():
                    storage_key = f"models/{safe_name}/v1/artifact.joblib"
                    try:
                        store.upload(alt_path, storage_key)
                        print(f"  Uploaded {model_name} -> {storage_key}")
                    except Exception as e:
                        print(f"  Warning: Failed to upload {model_name}: {e}")
                        storage_key = None

            mv = ModelVersion(
                model_id=model.id,
                version=1,
                storage_key=storage_key,
                metrics=entry.get("metrics", {}),
                is_current=True,
                properties={
                    "hyperparameters": entry.get("hyperparameters", {}),
                    "training_samples": entry.get("training_samples", 0),
                    "classes": entry.get("classes", []),
                },
            )
            session.add(mv)
            count += 1

    print(f"  Backfilled {count} models from registry.json.")


def main():
    print("=" * 60)
    print("Jubilee Registry Backfill")
    print("=" * 60)

    store = get_artifact_store()
    print(f"Artifact store: {type(store).__name__}")
    settings = get_settings()
    print(f"R2 enabled: {settings.r2_enabled}")

    print("\n[1/2] Backfilling datasets from catalog.json...")
    backfill_catalog()

    print("\n[2/2] Backfilling models from registry.json...")
    backfill_models()

    print("\nDone.")


if __name__ == "__main__":
    main()
