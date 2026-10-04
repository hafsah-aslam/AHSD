"""Read raw NF-v3 CSV (plain or zipped) without modifying it.

Numeric columns are read as float64 for timestamps (ms since epoch exceed
float32 precision) and float32 otherwise, to keep 20M-row datasets in memory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from . import schema


def _csv_member(path: Path) -> str | None:
    if path.suffix.lower() != ".zip":
        return None
    with zipfile.ZipFile(path) as z:
        # UQ BagIt packages also hold NetFlow_v3_Features.csv (feature descriptions); not data
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
                and not n.rsplit("/", 1)[-1].lower().startswith("netflow_v3_features")]
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


def _dtypes(header: list[str]) -> dict:
    dtypes = {}
    for c in header:
        if c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", schema.LABEL_MULTI):
            dtypes[c] = "category"
        elif c in (schema.TIME_START, schema.TIME_END):
            dtypes[c] = "float64"  # epoch ms exceed float32 precision
        elif c == schema.LABEL_BIN:
            dtypes[c] = "int8"
        else:
            dtypes[c] = "float32"
    return dtypes


def read_raw(path: Path, nrows: int | None = None, n_rows_hint: int | None = None,
             chunksize: int = 2_000_000) -> pd.DataFrame:
    """Load a raw NF-v3 file without modifying it.

    Parsed chunk by chunk into preallocated column arrays, so peak memory stays
    near the final frame size (27M-row files on a 15 GB machine). String
    columns become categoricals; labels are stripped of surrounding spaces.
    """
    path = Path(path)
    header = read_header(path)
    dtypes = _dtypes(header)
    member = _csv_member(path)
    kw = dict(dtype=dtypes, nrows=nrows, engine="c", chunksize=chunksize)
    num_cols = [c for c in header if dtypes[c] != "category"]
    cat_cols = [c for c in header if dtypes[c] == "category"]
    total = nrows if nrows is not None else n_rows_hint
    arrays = {c: np.empty(total, dtype=dtypes[c]) for c in num_cols} if total else None
    pieces = {c: [] for c in cat_cols}
    chunks, pos = [], 0
    f = zipfile.ZipFile(path).open(member) if member else open(path, "rb")
    with f:
        for ch in pd.read_csv(f, **kw):
            n = len(ch)
            if arrays is not None:
                if pos + n > total:
                    raise ValueError(f"{path}: more rows than the hint {total}")
                for c in num_cols:
                    arrays[c][pos:pos + n] = ch[c].to_numpy()
            else:
                chunks.append(ch[num_cols])
            for c in cat_cols:
                pieces[c].append(ch[c].astype("category"))
            pos += n
    if arrays is not None:
        if pos != total:
            raise ValueError(f"{path}: read {pos} rows, expected {total}")
        data = arrays
    else:
        cat = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=num_cols)
        data = {c: cat[c].to_numpy() for c in num_cols}
    for c in cat_cols:
        u = union_categoricals(pieces[c]) if pieces[c] else pd.Categorical([])
        if c == schema.LABEL_MULTI:
            stripped = [str(x).strip() for x in u.categories]
            if len(set(stripped)) == len(stripped):
                u = u.rename_categories(stripped)
            else:
                u = pd.Categorical(np.asarray(u).astype(str), categories=None).astype(str)
                u = pd.Categorical(np.char.strip(u))
        data[c] = u
        pieces[c] = None
    return pd.DataFrame({c: data[c] for c in header}, copy=False)


def numeric_feature_frame(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].to_numpy(dtype=np.float64, copy=True)
