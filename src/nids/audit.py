"""Dataset audit: shape, dtypes, NaN/inf counts, class counts, time range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema


def audit_frame(df: pd.DataFrame, dataset_id: str, source_file: str, source_sha256: str) -> dict:
    num = df.select_dtypes(include=[np.number])
    # column by column: a float64 copy of the whole frame would not fit in memory for 27M rows
    nan = np.array([int(np.isnan(num[c].to_numpy()).sum()) if num[c].dtype.kind == "f" else 0 for c in num.columns])
    inf = np.array([int(np.isinf(num[c].to_numpy()).sum()) if num[c].dtype.kind == "f" else 0 for c in num.columns])
    out = {
        "dataset": dataset_id,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_columns": int(df.shape[1]),
        "schema_check": schema.check_columns(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nan_counts": {c: int(v) for c, v in zip(num.columns, nan) if v},
        "inf_counts": {c: int(v) for c, v in zip(num.columns, inf) if v},
        "nan_total": int(nan.sum()),
        "inf_total": int(inf.sum()),
    }
    if schema.LABEL_MULTI in df:
        out["attack_class_counts"] = {str(k): int(v) for k, v in df[schema.LABEL_MULTI].value_counts().items()}
    if schema.LABEL_BIN in df:
        out["label_counts"] = {str(k): int(v) for k, v in df[schema.LABEL_BIN].value_counts().items()}
    if schema.LABEL_BIN in df and schema.LABEL_MULTI in df:
        benign_multi = df[schema.LABEL_MULTI].str.lower() == schema.BENIGN.lower()
        out["label_consistency_violations"] = int(((df[schema.LABEL_BIN] == 0) != benign_multi).sum())
    if schema.TIME_START in df:
        t = df[schema.TIME_START].to_numpy()
        out["timestamp_range_ms"] = [float(np.nanmin(t)), float(np.nanmax(t))]
        out["timestamp_span_hours"] = float((np.nanmax(t) - np.nanmin(t)) / 3.6e6)
        out["timestamp_nan"] = int(np.isnan(t).sum())
        out["timestamps_sorted_in_file"] = bool(np.all(np.diff(t) >= 0))
    if schema.TIME_END in df and schema.TIME_START in df:
        out["end_before_start"] = int((df[schema.TIME_END] < df[schema.TIME_START]).sum())
    return out
