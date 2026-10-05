#!/usr/bin/env python
"""Build the D5 (Lycos2017) source file (AMENDMENT_04 A4.3 + operational choices).

1. Extract the untouched archive data/raw/lycos/lycos-ids2017-master.zip (SHA-256 checked)
   into data/interim/lycos_work/.
2. Run the authors' labelling.py unmodified (it labels LycoSTand flows by timestamp
   windows and addresses).
3. Adapter: timestamp (µs) -> FLOW_START_MILLISECONDS (ms); label -> Attack, Label = label != "benign";
   src_addr/dst_addr/src_port/dst_port -> NF identifier names (dropped by the pipeline);
   flow_id removed (identifier).
4. Write data/raw/lycos/derived/Lycos2017-D5.zip (one CSV) and record SHA-256s of the
   archive, labelling.py and the derived file in data/MANIFEST.json (key D5).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.provenance import ROOT, now_iso, require_clean_tree, sha256_file, write_json  # noqa: E402

ARCHIVE = ROOT / "data/raw/lycos/lycos-ids2017-master.zip"
ARCHIVE_SHA = "7457f65587fe609fbe983672c12577bb9a07996e6920f8e4a183b8a24a51cf1e"
URL = "https://maupiti-git.univ-lemans.fr/lycos/lycos-ids2017/archive/master.zip"
WORK = ROOT / "data/interim/lycos_work"
OUT = ROOT / "data/raw/lycos/derived/Lycos2017-D5.zip"
RENAME = {"src_addr": "IPV4_SRC_ADDR", "dst_addr": "IPV4_DST_ADDR", "src_port": "L4_SRC_PORT",
          "dst_port": "L4_DST_PORT"}


def main():
    require_clean_tree("D5 source build")
    if sha256_file(ARCHIVE) != ARCHIVE_SHA:
        raise SystemExit("Lycos archive SHA-256 differs from AMENDMENT_04")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; raw-side files are never overwritten")
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(WORK)
    repo = WORK / "lycos-ids2017"
    for zp in sorted((repo / "pcap_lycos").glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            z.extractall(repo / "pcap_lycos")
    (repo / "python_logs").mkdir(exist_ok=True)
    (repo / "lycos-ids2017").mkdir(exist_ok=True)
    labelling_sha = sha256_file(repo / "labelling.py")
    # labelling.py asks interactively for the number of worker processes (input()); the answer
    # only sets the pool size, not the labels. Supplied on stdin and recorded in the manifest.
    n_workers = "4"
    r = subprocess.run([sys.executable, "labelling.py"], cwd=repo, capture_output=True, text=True,
                       input=n_workers + "\n")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/d5_labelling.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"labelling.py failed (rc {r.returncode}); see logs/d5_labelling.log")

    files = sorted((repo / "lycos-ids2017").glob("*.csv"))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    files = sorted(files, key=lambda p: days.index(p.name.split("-")[0]))
    if len(files) != 5:
        raise SystemExit(f"expected 5 labelled CSVs, found {files}")
    parts = []
    for f in files:
        df = pd.read_csv(f, encoding="ISO-8859-1", low_memory=False)
        df["source_day_file"] = f.name
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    del parts
    n_in = len(df)
    label_counts = {str(k): int(v) for k, v in df["label"].value_counts().items()}
    if "NeedLabel" in label_counts:
        raise SystemExit(f"unlabelled flows remain: {label_counts['NeedLabel']}")
    out = df.drop(columns=["flow_id", "source_day_file"]).rename(columns=RENAME)
    out.insert(0, "FLOW_START_MILLISECONDS", out.pop("timestamp").astype("float64") / 1000.0)
    lab = out.pop("label").astype(str)
    out["Label"] = (lab != "benign").astype("int8")
    out["Attack"] = lab.map(lambda s: "Benign" if s == "benign" else s)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        with z.open("Lycos2017-D5.csv", "w") as fh:
            out.to_csv(fh, index=False)
    tmp.rename(OUT)
    OUT.chmod(0o444)
    man_path = ROOT / "data/MANIFEST.json"
    man = json.loads(man_path.read_text())
    man["datasets"]["D5"] = {
        "name": "Lycos2017 (LYCOS-IDS2017)", "status": "present", "file": str(OUT.relative_to(ROOT / "data/raw")),
        "url": URL, "landing": "https://lycos-ids.univ-lemans.fr/download-lycos-ids2017.html",
        "archive_file": str(ARCHIVE.relative_to(ROOT / "data/raw")), "archive_sha256": ARCHIVE_SHA,
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "labelling_py_sha256": labelling_sha, "labelling_py_stdin_workers": n_workers, "sha256": sha256_file(OUT), "size_bytes": OUT.stat().st_size,
        "row_count": int(len(out)), "rows_before_adapter": n_in, "columns": list(out.columns),
        "n_columns": len(out.columns), "label_counts_labelling_py": label_counts,
        "adapter": {"timestamp": "us -> FLOW_START_MILLISECONDS (ms)", "label": "-> Attack ('benign' -> 'Benign'), Label = != benign",
                    "renamed_identifiers": RENAME, "removed": ["flow_id"]},
        "licence": "see https://lycos-ids.univ-lemans.fr/ (cite Rosay et al., WI-IAT 2021, doi 10.1145/3486622.3493973)",
        "download_date_utc": now_iso()}
    write_json(man_path, man)
    shutil.rmtree(WORK)
    print(f"D5 source: {len(out):,} rows, {len(out.columns)} columns; labels {label_counts}")


if __name__ == "__main__":
    main()
