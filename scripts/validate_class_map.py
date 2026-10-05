#!/usr/bin/env python
"""Validate configs/d1_class_map.yaml against the D1 audit and the UQ-published family counts.

Fails closed (exit 1) unless every file sub-label maps to exactly one family, no
family lists an unknown sub-label, and every family's summed file count equals
the UQ page count. Writes results/d1_class_map_validation.json.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.provenance import ROOT, provenance, sha256_file, write_json  # noqa: E402


def validate(map_path: Path, audit_path: Path, published_path: Path) -> dict:
    cmap = yaml.safe_load(open(map_path))["families"]
    file_counts = json.loads(audit_path.read_text())["attack_class_counts"]
    pub = json.loads(published_path.read_text())["counts"]["D1"]
    owner = {}
    errors = []
    for fam, subs in cmap.items():
        for s in subs:
            if s in owner:
                errors.append(f"sub-label {s!r} mapped to both {owner[s]!r} and {fam!r}")
            owner[s] = fam
    errors += [f"file sub-label {s!r} not mapped" for s in file_counts if s not in owner]
    errors += [f"mapped sub-label {s!r} absent from file" for s in owner if s not in file_counts]
    pub_ci = {k.lower(): (k, v) for k, v in pub.items()}
    rows = {}
    for fam, subs in cmap.items():
        got = sum(file_counts.get(s, 0) for s in subs)
        ref = pub.get(fam, pub_ci.get(fam.lower(), (None, None))[1])
        rows[fam] = {"sub_labels": {s: file_counts.get(s, 0) for s in subs}, "file_sum": got, "uq_page": ref,
                     "match": got == ref}
        if got != ref:
            errors.append(f"family {fam!r}: file sum {got} != UQ page {ref}")
    errors += [f"UQ family {k!r} not in map" for k in pub if k not in cmap]
    return {"ok": not errors, "errors": errors, "families": rows,
            "map_sha256": sha256_file(map_path), "audit": str(audit_path.relative_to(ROOT))}


def main():
    res = validate(ROOT / "configs/d1_class_map.yaml", ROOT / "audit/D1.json",
                   ROOT / "configs/uq_published_class_counts_v3.json")
    res["provenance"] = provenance()
    write_json(ROOT / "results/d1_class_map_validation.json", res)
    for fam, r in res["families"].items():
        print(f"{fam:14s} file {r['file_sum']:>12,}  UQ {r['uq_page']:>12,}  {'OK' if r['match'] else 'MISMATCH'}")
    if not res["ok"]:
        print("\n".join(res["errors"]))
        sys.exit(1)
    print("class map valid")


if __name__ == "__main__":
    main()
