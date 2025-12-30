"""transform_apply - Apply ordered transforms to a dataset with full auditing."""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Type, Union

import pandas as pd

# Add types to path
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from jtypes.transformations import (BaseTransform, LineageStep,
                                    TransformApplyResult, TransformAudit,
                                    TransformLineage, TransformWarning)

# Import utilities
try:
    from ..utils import (build_schema_from_dataframe, generate_unique_id,
                         get_registered_dataset, register_dataset,
                         utc_timestamp)
except ImportError:
    from utils import (build_schema_from_dataframe, generate_unique_id,
                       get_registered_dataset, register_dataset, utc_timestamp)

# Import individual transforms directly (avoid circular import with __init__)
from .add_column import AddColumnTransform
from .cast import CastTransform
from .drop import DropTransform
from .parse_datetime import ParseDatetimeTransform
from .rename import RenameTransform
from .select import SelectTransform

# Local registry (avoids circular import)
_REGISTRY: dict[str, Type[BaseTransform]] = {
    "select": SelectTransform,
    "drop": DropTransform,
    "rename": RenameTransform,
    "cast": CastTransform,
    "parse_datetime": ParseDatetimeTransform,
    "add_column": AddColumnTransform,
}


def _get_transform(op_def: dict) -> BaseTransform:
    """Create transform from op definition."""
    if not isinstance(op_def, dict) or "op" not in op_def:
        raise ValueError("Op definition must be dict with 'op' key")
    
    op_name = op_def["op"]
    if op_name not in _REGISTRY:
        raise ValueError(f"Unknown op: '{op_name}'. Available: {list(_REGISTRY.keys())}")
    
    params = {k: v for k, v in op_def.items() if k != "op"}
    try:
        return _REGISTRY[op_name](**params)
    except TypeError as e:
        raise ValueError(f"Invalid params for '{op_name}': {e}")


def _resolve_dataset(dataset_ref: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Resolve dataset reference to DataFrame."""
    if isinstance(dataset_ref, pd.DataFrame):
        return dataset_ref
    
    df = get_registered_dataset(dataset_ref)
    if df is not None:
        return df
    
    try:
        try:
            from ..data_loader import dataset_get
        except ImportError:
            from data_loader import dataset_get
        result = dataset_get(asset_id=dataset_ref, limit=-1, include_stats=False)
        return pd.DataFrame(result.data)
    except Exception:
        pass
    
    raise ValueError(f"Could not resolve: '{dataset_ref}'")


def transform_apply(
    dataset_ref: Union[str, pd.DataFrame],
    ops: list[dict],
    dry_run: bool = False,
    return_lineage: bool = True,
    return_intermediates: bool = False,
    stop_on_error: bool = True,
    register_result: bool = True,
) -> TransformApplyResult:
    """Apply ordered transforms to dataset. Returns auditable result with lineage."""
    start = time.perf_counter()
    
    # Resolve input
    try:
        df = _resolve_dataset(dataset_ref)
        source_ref = dataset_ref if isinstance(dataset_ref, str) else "input_dataframe"
    except ValueError as e:
        return TransformApplyResult("", False, 0, len(ops), [], error=str(e),
                                    execution_time_ms=(time.perf_counter() - start) * 1000)
    
    audits, lineage_steps, intermediates = [], [], {}
    warnings_total, current_df, current_ref = 0, df.copy(), source_ref
    pipeline_id = generate_unique_id("tfm", *[op.get("op", "") for op in ops])
    
    # Parse all ops first
    transforms = []
    for i, op_def in enumerate(ops):
        try:
            transforms.append(_get_transform(op_def))
        except ValueError as e:
            audits.append(TransformAudit(op_def.get("op", "unknown"), op_def, len(current_df), len(current_df),
                                         list(current_df.columns), list(current_df.columns), success=False, error=str(e)))
            return TransformApplyResult("", False, i, len(ops), audits, error=f"Validation error at step {i}: {e}",
                                        execution_time_ms=(time.perf_counter() - start) * 1000)
    
    # Dry run
    if dry_run:
        for transform in transforms:
            try:
                warnings = transform.validate(current_df)
                audits.append(TransformAudit(transform.op_name, transform.get_params(), len(current_df), len(current_df),
                                             list(current_df.columns), list(current_df.columns), warnings=warnings, success=True))
                warnings_total += len(warnings)
            except ValueError as e:
                audits.append(TransformAudit(transform.op_name, transform.get_params(), len(current_df), len(current_df),
                                             list(current_df.columns), list(current_df.columns), success=False, error=str(e)))
                if stop_on_error:
                    break
        
        return TransformApplyResult("[dry_run]", all(a.success for a in audits), len(audits), len(ops), audits,
                                    final_schema=build_schema_from_dataframe(current_df), final_rows=len(current_df),
                                    warnings_total=warnings_total, execution_time_ms=(time.perf_counter() - start) * 1000)
    
    # Execute
    for i, transform in enumerate(transforms):
        step_ref = f"{pipeline_id}_step{i}"
        try:
            result = transform.execute(current_df)
            audits.append(result.audit)
            warnings_total += len(result.audit.warnings)
            
            if return_lineage:
                lineage_steps.append(LineageStep(i, transform.op_name, transform.get_params(), current_ref, step_ref, utc_timestamp()))
            if return_intermediates:
                intermediates[step_ref] = result.df.copy()
            
            current_df, current_ref = result.df, step_ref
            
        except Exception as e:
            audits.append(TransformAudit(transform.op_name, transform.get_params(), len(current_df), len(current_df),
                                         list(current_df.columns), list(current_df.columns), success=False, error=str(e)))
            if stop_on_error:
                return TransformApplyResult("", False, i, len(ops), audits, error=f"Error at step {i}: {e}",
                                            execution_time_ms=(time.perf_counter() - start) * 1000)
    
    output_ref = generate_unique_id("ds", pipeline_id)
    if register_result:
        register_dataset(output_ref, current_df)
    
    lineage = None
    if return_lineage:
        lineage = TransformLineage(pipeline_id, source_ref, output_ref, lineage_steps,
                                   total_execution_time_ms=(time.perf_counter() - start) * 1000)
    
    return TransformApplyResult(output_ref, True, len(transforms), len(ops), audits, lineage,
                                intermediates if return_intermediates else None,
                                build_schema_from_dataframe(current_df), len(current_df), warnings_total,
                                (time.perf_counter() - start) * 1000)
