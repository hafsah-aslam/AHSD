"""Read raw NF-v3 CSV (plain or zipped) without modifying it.

Numeric columns are read as float64 for timestamps (ms since epoch exceed
float32 precision) and float32 otherwise, to keep 20M-row datasets in memory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema


def _csv_member(path: Path) -> str | None:
    if path.suffix.lower() != ".zip":
        return None
    with zipfile.ZipFile(path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
    if len(csvs) != 1:
        raise ValueError(f"{path}: expected exactly one CSV inside the zip, found {csvs}")
    return csvs[0]


def find_raw_file(raw_dir: Path, pattern: str) -> Path:
    hits = sorted(p for p in Path(raw_dir).glob(pattern) if p.is_file())
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one file matching {pattern!r} in {raw_dir}, found {hits}")
    return hits[0]


def read_header(path: Path) -> list[str]:
    path = Path(path)
    member = _csv_member(path)
    if member:
        with zipfile.ZipFile(path) as z, z.open(member) as f:
            return f.readline().decode().strip().split(",")
    with open(path) as f:
        return f.readline().strip().split(",")


def read_raw(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a raw NF-v3 file. Labels stay as read; numerics are downcast."""
    path = Path(path)
    header = read_header(path)
    dtypes = {}
    for c in header:
        if c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", schema.LABEL_MULTI):
            dtypes[c] = "category"
        elif c in (schema.TIME_START, schema.TIME_END):
            dtypes[c] = "float64"
        elif c == schema.LABEL_BIN:
            dtypes[c] = "int8"
        else:
            dtypes[c] = "float32"
    member = _csv_member(path)
    kw = dict(dtype=dtypes, nrows=nrows, engine="c", low_memory=False)
    if member:
        with zipfile.ZipFile(path) as z, z.open(member) as f:
            df = pd.read_csv(f, **kw)
    else:
        df = pd.read_csv(path, **kw)
    if schema.LABEL_MULTI in df:
        cat = df[schema.LABEL_MULTI].astype("category")
        df[schema.LABEL_MULTI] = cat.cat.rename_categories([str(c).strip() for c in cat.cat.categories]) \
            if len(set(str(c).strip() for c in cat.cat.categories)) == len(cat.cat.categories) \
            else cat.astype(str).str.strip().astype("category")
    return df


def numeric_feature_frame(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].to_numpy(dtype=np.float64, copy=True)
