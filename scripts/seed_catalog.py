#!/usr/bin/env python3
"""Seed Postgres `datasets` from catalog.json: CSVs → Parquet on R2 + DB metadata.

Runtime code loads tabular data via Postgres `storage_key` + parquet only
(`get_registered_dataset` / `_download_parquet_from_r2`). This script must
therefore upload **parquet** objects, not raw CSV, and must **not** insert rows
without a successful R2 upload (avoids metadata-only ghost datasets).

SQL and HuggingFace Python catalog entries are skipped here; materialize those
elsewhere or extend this script if needed.

Usage (from repo root, with local `datasets/` CSVs and R2 env configured):

    python scripts/seed_catalog.py
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "datasets"
CATALOG_PATH = DATASETS_DIR / "catalog.json"


def _catalog_parquet_key(file_key: str) -> str:
    """Stable R2 key under datasets/catalog/*.parquet (no spaces / odd chars)."""
    slug = file_key.replace("/", "_").replace(" ", "_")
    slug = re.sub(r"[^\w\-.]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "dataset"
    if not slug.endswith(".parquet"):
        slug = f"{slug}.parquet"
    return f"datasets/catalog/{slug}"


def main():
    import pandas as pd
    from sqlalchemy import or_

    from backend.shared.database import get_db_session
    from backend.shared.models import Dataset
    from backend.shared.artifact_store import get_artifact_store

    if not CATALOG_PATH.exists():
        print(f"No catalog found at {CATALOG_PATH}")
        return

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    entries = catalog.get("datasets", catalog if isinstance(catalog, list) else [])
    if not entries:
        print("No dataset entries found in catalog.")
        return

    store = get_artifact_store()
    inserted = 0
    skipped = 0
    uploaded = 0

    with get_db_session() as session:
        for entry in entries:
            file_key = entry.get("file")
            name = entry.get("name", file_key or "unknown")
            fmt = (entry.get("format") or "").strip()

            if not file_key:
                print(f"  SKIP (no file): {name}")
                skipped += 1
                continue

            if fmt.upper() != "CSV" and not str(file_key).lower().endswith(".csv"):
                print(f"  SKIP (not CSV — no parquet seed): {name} [{file_key}] ({fmt or 'unknown format'})")
                skipped += 1
                continue

            local_path = DATASETS_DIR / file_key
            if not local_path.exists():
                print(f"  SKIP (missing file on disk): {local_path}")
                skipped += 1
                continue

            existing = (
                session.query(Dataset)
                .filter(Dataset.name == name)
                .filter(
                    or_(
                        Dataset.properties["catalog_file"].astext == file_key,
                        Dataset.properties["file"].astext == file_key,
                    )
                )
                .first()
            )
            if existing:
                sk = (existing.properties or {}).get("storage_key") or ""
                if sk.endswith(".parquet"):
                    print(f"  EXISTS: {name} [{file_key}]")
                    skipped += 1
                    continue

            storage_key = _catalog_parquet_key(file_key)
            tmp_path = None
            try:
                df = pd.read_csv(local_path)
                row_count = len(df)
                columns = list(df.columns)
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = tmp.name
                df.to_parquet(tmp_path, index=False)
                store.upload(Path(tmp_path), storage_key)
                print(f"  R2 OK : {storage_key} ({row_count} rows)")
                uploaded += 1
            except Exception as exc:
                print(f"  R2 ERR: {file_key} — {exc}")
                skipped += 1
                continue
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            properties = {
                "catalog_file": file_key,
                "format": "parquet",
                "description": entry.get("description", ""),
                "use_case": entry.get("use_case", ""),
                "columns": columns,
                "row_count": row_count,
                "storage_key": storage_key,
            }
            if entry.get("source"):
                properties["source"] = entry["source"]

            if existing:
                existing.properties = {**(existing.properties or {}), **properties}
                existing.source_type = "local"
                print(f"  UPDATE: {name} [{file_key}] → parquet metadata")
            else:
                session.add(
                    Dataset(
                        name=name,
                        source_type="local",
                        properties=properties,
                    )
                )
                print(f"  INSERT: {name} [{file_key}]")
            inserted += 1

    print(f"\nDone — rows written/updated: {inserted}, skipped: {skipped}, parquet uploads: {uploaded}")


if __name__ == "__main__":
    main()
