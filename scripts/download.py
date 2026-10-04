#!/usr/bin/env python
"""Download the four UQ NetFlow v3 datasets into data/raw/ and write data/MANIFEST.json.

Sources (spec §3): https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (v3 section)
and the UQ Research Data records. Direct file URLs go in configs/download.yaml;
they are left empty until checked by hand against the UQ page.

  python scripts/download.py            # download every dataset with a URL, then register
  python scripts/download.py --register # hash files already placed in data/raw/ by hand
  python scripts/download.py --smoke    # check the manifest logic on a tiny local file

If a URL answers with an HTML page (licence/terms, login) the script stops and
asks for a manual download. Raw files are never modified; Kaggle mirrors are
never used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import io, schema  # noqa: E402
from nids.provenance import ROOT, now_iso, sha256_file, write_json  # noqa: E402


def count_rows(path: Path) -> int:
    member = io._csv_member(path)
    n = 0
    if member:
        with zipfile.ZipFile(path) as z, z.open(member) as f:
            for _ in f:
                n += 1
    else:
        with open(path, "rb") as f:
            for _ in f:
                n += 1
    return max(n - 1, 0)


def register(raw_dir: Path, datasets: dict, urls: dict, manifest_path: Path, existing: dict) -> dict:
    manifest = {"generated_utc": now_iso(), "datasets": {}}
    for ds, d in datasets.items():
        hits = sorted(p for p in raw_dir.glob(d["raw_glob"]) if p.is_file())
        if not hits:
            manifest["datasets"][ds] = {"name": d["name"], "status": "missing",
                                        "action": f"place the file in {raw_dir.relative_to(ROOT) if raw_dir.is_relative_to(ROOT) else raw_dir} (pattern {d['raw_glob']})"}
            continue
        if len(hits) > 1:
            raise SystemExit(f"{ds}: several raw files match {d['raw_glob']}: {hits}")
        p = hits[0]
        prev = existing.get("datasets", {}).get(ds, {})
        sha = sha256_file(p)
        if prev.get("sha256") and prev["sha256"] != sha:
            raise SystemExit(f"{ds}: {p.name} changed since it was registered (raw files are immutable)")
        header = io.read_header(p)
        manifest["datasets"][ds] = {
            "name": d["name"], "status": "present", "file": p.name, "url": urls.get(ds) or prev.get("url"),
            "sha256": sha, "size_bytes": p.stat().st_size, "row_count": count_rows(p),
            "columns": header, "schema_check": schema.check_columns(header),
            "download_date_utc": prev.get("download_date_utc") or now_iso(),
        }
        print(f"{ds}: {p.name} rows={manifest['datasets'][ds]['row_count']} sha256={sha[:12]}…")
    write_json(manifest_path, manifest)
    return manifest


def download(url: str, dest_dir: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "nids-research-download/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise SystemExit(f"{url} returned an HTML page (terms/login?). Download it manually into {dest_dir}.")
        name = Path(urllib.parse.urlparse(r.geturl()).path).name or "download.bin"
        cd = r.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            name = cd.split("filename=")[-1].strip('"; ')
        dest = dest_dir / name
        if dest.exists():
            print(f"exists, not overwritten: {dest}")
            return dest
        with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tmp:
            shutil.copyfileobj(r, tmp, length=1 << 22)
        Path(tmp.name).rename(dest)
    dest.chmod(0o444)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--urls", default="configs/download.yaml")
    ap.add_argument("--register", action="store_true", help="only hash files already in data/raw")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip datasets already present (default behaviour)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    urls = (yaml.safe_load(open(ROOT / a.urls)) or {}).get("urls", {}) if (ROOT / a.urls).exists() else {}

    if a.smoke:
        from nids.synthetic import generate
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            generate(2000, hours=2).to_csv(td / "NF-CSE-CIC-IDS2018-v3.csv", index=False)
            m = register(td, {"D1": cfg["datasets"]["D1"]}, {}, td / "MANIFEST.json", {})
            assert m["datasets"]["D1"]["row_count"] == 2000, m
            assert not m["datasets"]["D1"]["schema_check"]["missing"], m
        print("download --smoke OK")
        return

    raw_dir = ROOT / cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data" / "MANIFEST.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not a.register:
        for ds, d in cfg["datasets"].items():
            if any(raw_dir.glob(d["raw_glob"])):
                print(f"{ds}: already in {raw_dir}, skipped")
                continue
            url = urls.get(ds)
            if not url:
                print(f"{ds}: no verified URL in {a.urls}; download {d['name']} manually from the UQ page into {raw_dir}")
                continue
            print(f"{ds}: downloading {url}")
            try:
                download(url, raw_dir)
            except (urllib.error.URLError, OSError) as e:
                print(f"{ds}: download failed ({e}). Download manually into {raw_dir}.")
    register(raw_dir, cfg["datasets"], urls, manifest_path, existing)


if __name__ == "__main__":
    main()
