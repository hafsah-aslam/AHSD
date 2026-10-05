#!/usr/bin/env python
"""Backfill code_commit for the P1 results written by PID 5266 (before code_commit existed)."""
import json, glob, subprocess, sys
ROOT = "/home/user/AHSD"
CODE = "f557bcd4602719b6a160e46b4a9ad1b503a5e6fc"
PATHS = ["src", "scripts/run_loaco.py", "configs/p1_pilot.yaml", "configs/data.yaml"]
START, LOG = "2026-10-05T00:26:14+00:00", "logs/p1_run.log"
def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
n = 0
for p in sorted(glob.glob(f"{ROOT}/results/p1/**/*.json", recursive=True)):
    if p.endswith("summary.json"):
        continue
    r = json.load(open(p))
    pv = r["provenance"]
    if "code_commit" in pv:
        continue
    head = pv["git_hash"]
    assert not head.endswith("-dirty"), p
    assert git("diff", "--stat", CODE, head, "--", *PATHS) == "", f"{p}: {head} differs from {CODE}"
    assert pv["timestamp_utc"] >= START, p
    pv["results_head"] = pv.pop("git_hash")
    pv["code_commit"] = CODE
    pv["code_dirty"] = False
    pv["code_commit_captured_utc"] = START
    pv["code_commit_backfill"] = {
        "backfilled": True, "process": "PID 5266, started 2026-10-05T00:26:14Z",
        "method": "the field did not exist when this run was written; code_commit is the last commit before process "
                  "start (f557bcd); results_head (the recorded git_hash) was verified to have an empty diff over "
                  + ", ".join(PATHS) + " against it, and was not dirty"}
    json.dump(r, open(p, "w"), indent=2)
    n += 1
print("backfilled", n)
