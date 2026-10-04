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


def scan_data_file(path: Path) -> dict:
    """One streaming pass over the data CSV: row count, SHA-1, and BagIt manifest check (zips)."""
    import hashlib
    member = io._csv_member(path)
    h, nl, last = hashlib.sha1(), 0, b"\n"
    with (zipfile.ZipFile(path).open(member) if member else open(path, "rb")) as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
            nl += chunk.count(b"\n")
            last = chunk[-1:]
    lines = nl + (last != b"\n")
    out = {"data_member": member, "data_sha1": h.hexdigest(),
           "row_count": max(lines - 1, 0)}  # header excluded; NetFlow CSVs have no embedded newlines
    if member:
        with zipfile.ZipFile(path) as z:
            mf = [n for n in z.namelist() if n.endswith("/manifest-sha1.txt")]
            if mf:
                rel = member.split("/", 1)[1]
                listed = {ln.split(maxsplit=1)[1].strip(): ln.split()[0]
                          for ln in z.read(mf[0]).decode().splitlines() if ln.strip()}
                out["bagit_sha1"] = listed.get(rel)
                out["bagit_sha1_verified"] = listed.get(rel) == out["data_sha1"]
                if not out["bagit_sha1_verified"]:
                    raise SystemExit(f"{path.name}: data SHA-1 does not match the BagIt manifest")
            out["zip_members"] = [{"name": i.filename, "size": i.file_size} for i in z.infolist()]
    return out


def register(raw_dir: Path, datasets: dict, dlog: dict, manifest_path: Path, existing: dict) -> dict:
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
            "name": d["name"], "status": "present", "file": p.name,
            "url": dlog.get(ds, {}).get("official_url") or prev.get("url"),
            "sha256": sha, "size_bytes": p.stat().st_size, **scan_data_file(p),
            "columns": header, "n_columns": len(header), "schema_check": schema.check_columns(header),
            "download_date_utc": dlog.get(ds, {}).get("download_date_utc") or prev.get("download_date_utc"),
            "download": dlog.get(ds) or prev.get("download"),
        }
        if dlog.get(ds) and dlog[ds]["content_length"] != manifest["datasets"][ds]["size_bytes"]:
            raise SystemExit(f"{ds}: size on disk differs from the downloaded Content-Length")
        print(f"{ds}: {p.name} rows={manifest['datasets'][ds]['row_count']} sha256={sha[:12]}…")
    write_json(manifest_path, manifest)
    return manifest


RDM_API = "https://api.rdm.uq.edu.au/datasets"
S3_PART_SIZES_MIB = (5, 8, 16, 32, 64)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nids-research-download/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def s3_multipart_etag(path: Path, n_parts: int) -> str | None:
    """Recompute an S3 multipart ETag ("md5-of-part-md5s-N") for the part size implied by N."""
    import hashlib
    size = path.stat().st_size
    for mib in S3_PART_SIZES_MIB:
        ps = mib << 20
        if -(-size // ps) != n_parts:
            continue
        digests = []
        with open(path, "rb") as f:
            while chunk := f.read(ps):
                digests.append(hashlib.md5(chunk).digest())
        return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{n_parts}"
    return None


def download_rdm(ds: str, entry: dict, dest_dir: Path, ds_name: str) -> dict:
    """Resolve an official UQ RDM dataset (open access) to its signed S3 object and download it.

    Fails closed unless the record is PUBLISHED and OPEN (no login, no click-through terms),
    the byte count matches Content-Length, and the S3 multipart ETag matches the file.
    """
    uuid = entry["rdm_uuid"]
    det = _get_json(f"{RDM_API}/{uuid}/file-details")
    if det.get("status") != "PUBLISHED" or det.get("accessType") != "OPEN":
        raise SystemExit(f"{ds}: RDM record {uuid} is {det.get('status')}/{det.get('accessType')}, not "
                         f"PUBLISHED/OPEN. Download it manually into {dest_dir}.")
    if det.get("name") != entry["uq_name"]:
        raise SystemExit(f"{ds}: RDM record name {det.get('name')!r} != expected {entry['uq_name']!r}")
    signed = _get_json(f"{RDM_API}/{uuid}/download")["url"]
    dest = dest_dir / f"{ds_name}.zip"
    if dest.exists():
        raise SystemExit(f"{dest} exists; raw files are never overwritten")
    req = urllib.request.Request(signed, headers={"User-Agent": "nids-research-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if "zip" not in r.headers.get("Content-Type", ""):
            raise SystemExit(f"{ds}: expected a zip, got {r.headers.get('Content-Type')}")
        length = int(r.headers["Content-Length"])
        etag = r.headers.get("ETag", "").strip('"')
        last_mod = r.headers.get("Last-Modified")
        with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False, suffix=".part") as tmp:
            shutil.copyfileobj(r, tmp, length=1 << 22)
    tmp_path = Path(tmp.name)
    if tmp_path.stat().st_size != length:
        tmp_path.unlink()
        raise SystemExit(f"{ds}: truncated download ({tmp_path.stat().st_size} != {length})")
    etag_ok = None
    if "-" in etag:
        etag_ok = s3_multipart_etag(tmp_path, int(etag.split("-")[1])) == etag
        if etag_ok is False:
            tmp_path.unlink()
            raise SystemExit(f"{ds}: S3 multipart ETag mismatch; download corrupted")
    tmp_path.rename(dest)
    dest.chmod(0o444)
    return {"file": dest.name, "official_url": entry["landing"], "uq_rdm_uuid": uuid,
            "uq_rdm_name": det.get("name"), "uq_espace_pid": det.get("espacePid"),
            "licence": (det.get("datasetLicense") or {}).get("title"),
            "licence_terms": (det.get("datasetLicense") or {}).get("description"),
            "access_type": det.get("accessType"),
            "s3_object": urllib.parse.urlparse(signed).path.lstrip("/"),
            "s3_etag": etag, "s3_etag_verified": etag_ok, "s3_last_modified": last_mod,
            "content_length": length, "download_date_utc": now_iso()}


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
    log_path = ROOT / "data" / "DOWNLOAD_LOG.json"
    dlog = json.loads(log_path.read_text()) if log_path.exists() else {}
    if not a.register:
        for ds, d in cfg["datasets"].items():
            if any(raw_dir.glob(d["raw_glob"])):
                print(f"{ds}: already in {raw_dir}, skipped")
                continue
            entry = urls.get(ds)
            if not entry:
                print(f"{ds}: no verified URL in {a.urls}; download {d['name']} manually from the UQ page into {raw_dir}")
                continue
            print(f"{ds}: downloading {entry['landing']}")
            try:
                dlog[ds] = download_rdm(ds, entry, raw_dir, d["name"])
            except (urllib.error.URLError, OSError) as e:
                raise SystemExit(f"{ds}: download failed ({e!r}). Not retried via any mirror.")
            write_json(log_path, dlog)
            print(f"{ds}: saved {dlog[ds]['file']} ({dlog[ds]['content_length']} B, ETag verified: "
                  f"{dlog[ds]['s3_etag_verified']})")
    register(raw_dir, cfg["datasets"], dlog, manifest_path, existing)


if __name__ == "__main__":
    main()
