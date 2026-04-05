import json
import re
import tempfile
import unicodedata
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backend.catalog import repository
from backend.shared.artifact_store import get_artifact_store
from backend.shared.database import get_db_session
from backend.shared.models import Dataset as DatasetModel
from backend.shared.settings import get_settings


class DatasetUploadValidationError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class DatasetUploadConflictError(Exception):
    def __init__(self, detail: str = "A dataset with this name already exists"):
        self.detail = detail
        super().__init__(detail)


def _slugify_dataset_name(raw: str, *, max_len: int = 256) -> str:
    s = unicodedata.normalize("NFKD", raw)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    if not s:
        s = "dataset"
    return s[:max_len]


def _extension(filename: str | None) -> str:
    if not filename or not filename.strip():
        return ""
    return Path(filename).suffix.lower()


def _read_csv_auto_encoding(path: Path) -> pd.DataFrame:
    """Try common encodings; byte 0xa3 etc. often means Latin-1/Windows-1252, not UTF-8."""
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last: UnicodeDecodeError | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
            continue
    raise DatasetUploadValidationError(
        f"Could not decode CSV (tried utf-8, utf-8-sig, cp1252, latin-1): {last}"
    ) from last


def _read_dataframe(path: Path, ext: str) -> pd.DataFrame:
    try:
        if ext == ".csv":
            return _read_csv_auto_encoding(path)
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
    except DatasetUploadValidationError:
        raise
    except Exception as e:
        raise DatasetUploadValidationError(f"Could not parse file: {e}") from e
    raise DatasetUploadValidationError("Unsupported file format")


def upload_user_dataset(
    file_content: bytes,
    filename: str | None,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, object]:
    if not file_content:
        raise DatasetUploadValidationError("Empty file")

    ext = _extension(filename)
    if ext not in (".csv", ".parquet", ".pq"):
        raise DatasetUploadValidationError(
            "Unsupported extension; allowed: .csv, .parquet"
        )

    if name and name.strip():
        base_name = _slugify_dataset_name(name.strip())
    else:
        stem = Path(filename or "dataset").stem or "dataset"
        base_name = _slugify_dataset_name(stem)

    tmp_suffix = ".csv" if ext == ".csv" else ".parquet"
    with tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False) as tmp_in:
        tmp_in.write(file_content)
        tmp_in_path = Path(tmp_in.name)

    parquet_path: Path | None = None
    try:
        df = _read_dataframe(tmp_in_path, ext)
        if df.empty or len(df) == 0:
            raise DatasetUploadValidationError("Dataset has no rows")

        row_count = int(len(df))
        columns = [str(c) for c in df.columns.tolist()]

        with get_db_session() as session:
            if repository.dataset_name_exists(session, base_name):
                raise DatasetUploadConflictError()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_out:
            parquet_path = Path(tmp_out.name)
        df.to_parquet(parquet_path, index=False)

        key = f"datasets/uploaded/{uuid.uuid4()}.parquet"
        store = get_artifact_store()
        store.upload(parquet_path, key)

        desc = (description or "").strip()
        orig_name = Path(filename).name if filename else ""

        properties: dict[str, object] = {
            "storage_key": key,
            "format": "parquet",
            "row_count": row_count,
            "columns": columns,
            "description": desc,
            "original_filename": orig_name,
        }

        with get_db_session() as session:
            if repository.dataset_name_exists(session, base_name):
                try:
                    store.delete(key)
                except Exception:
                    pass
                raise DatasetUploadConflictError()
            row = repository.insert_dataset(
                session,
                name=base_name,
                source_type="uploaded",
                properties=properties,
            )
            return repository.dataset_to_dict(row)
    finally:
        try:
            tmp_in_path.unlink(missing_ok=True)
        except OSError:
            pass
        if parquet_path is not None:
            try:
                parquet_path.unlink(missing_ok=True)
            except OSError:
                pass


def get_datasets(include_derived: bool = False) -> list[dict[str, object]]:
    return repository.get_datasets(include_derived=include_derived)


def _read_parquet_head(path: Path, limit: int) -> pd.DataFrame:
    """Read at most ``limit`` rows without loading the full file into memory."""
    pf = pq.ParquetFile(path)
    tables: list = []
    n = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        rem = limit - n
        if rem <= 0:
            break
        if t.num_rows <= rem:
            tables.append(t)
            n += t.num_rows
        else:
            tables.append(t.slice(0, rem))
            break
    if not tables:
        return pd.DataFrame()
    out = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return out.to_pandas()


def _read_csv_head(path: Path, limit: int) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last: UnicodeDecodeError | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=limit)
        except UnicodeDecodeError as e:
            last = e
            continue
    raise ValueError(
        f"Could not decode CSV (tried utf-8, utf-8-sig, cp1252, latin-1): {last}"
    ) from last


def _dataframe_preview_payload(df: pd.DataFrame) -> tuple[list[str], list[dict[str, object]]]:
    cols = [str(c) for c in df.columns.tolist()]
    records = json.loads(
        df.to_json(orient="records", date_format="iso", default_handler=str)
    )
    return cols, records


def get_dataset_preview(ref: str, limit: int = 25) -> dict[str, object]:
    """Load up to ``limit`` rows from a dataset's backing file or object storage."""
    limit = max(1, min(limit, 100))
    with get_db_session() as session:
        row = session.query(DatasetModel).filter_by(name=ref).first()
        if not row:
            raise LookupError("Dataset not found")
        props = row.properties or {}
        file_prop = props.get("file")
        storage_key = props.get("storage_key")

    if not file_prop and not storage_key:
        raise LookupError("No previewable data for this dataset")

    tmp_path: Path | None = None
    try:
        if file_prop:
            path = get_settings().datasets_dir / str(file_prop)
            if not path.is_file():
                raise LookupError("Dataset file not found on server")
            ext = path.suffix.lower()
            if ext == ".csv":
                df = _read_csv_head(path, limit)
            elif ext in (".parquet", ".pq"):
                df = _read_parquet_head(path, limit)
            else:
                raise LookupError("Unsupported dataset file format")
        else:
            assert storage_key is not None
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            store = get_artifact_store()
            store.download(str(storage_key), tmp_path)
            df = _read_parquet_head(tmp_path, limit)

        columns, rows = _dataframe_preview_payload(df)
        return {
            "name": ref,
            "limit": limit,
            "returned": len(rows),
            "columns": columns,
            "rows": rows,
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def get_models() -> list[dict[str, str]]:
    return repository.get_models()


def get_trained_models() -> dict[str, object]:
    return repository.get_trained_models()
