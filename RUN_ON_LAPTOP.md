# Running HoNE on your own computer

This folder is a git checkout of the HoNE project (code, configs, tests, pre-registration, all result
files, release splits and the paper). The raw datasets are **not** included (licence and size); step 2
downloads them from the official sources.

There are two ways to use it:

- **A. Check the paper against the stored results (minutes, no datasets needed).**
- **B. Re-run everything from the raw data (about one day on a 4-core laptop).**

## 1. Set up Python (both paths)

Python 3.11 is recommended (3.10–3.12 work).

```bash
cd AHSD
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # numpy, pandas, pyarrow, scipy, scikit-learn, torch (CPU), xgboost, matplotlib, pyyaml, pytest
python -m pytest -q                  # unit and regression tests
```

Keep the folder a git checkout (the `.git` directory must stay). The experiment scripts refuse to start
on a tree with uncommitted changes to tracked files, so that every result records the exact code commit.
Do your own edits on a new branch and commit them before running anything.

## A. Check the paper against the stored results

```bash
python paper/hone/check_numbers.py   # 42 checks: every number quoted in the paper vs results/final_v2
python paper/hone/gen_appendix.py    # rebuilds the appendix tables from the result JSON
cd paper/hone && pdflatex main.tex && pdflatex main.tex   # needs a LaTeX installation (TeX Live / MiKTeX)
```

To show author names instead of "Anonymous", set `\anonymousfalse` on line 11 of `paper/hone/main.tex`.

## B. Re-run everything from the raw data

Hardware used for the paper: 4 CPU cores, 16 GB RAM, no GPU. Plan for about 20 GB of free disk.
Approximate times on that hardware: data preparation 2–4 h, final run 10.8 h.

### 2. Download the datasets (official sources only)

| ID | Dataset | Save as | Source |
|---|---|---|---|
| D1 | NF-CSE-CIC-IDS2018-v3 | `data/raw/NF-CSE-CIC-IDS2018-v3.zip` | https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22 |
| D2 | NF-UNSW-NB15-v3 | `data/raw/NF-UNSW-NB15-v3.zip` | https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f |
| D3 | NF-ToN-IoT-v3 | `data/raw/NF-ToN-IoT-v3.zip` | https://rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4 |
| D4 | NF-BoT-IoT-v3 | `data/raw/NF-BoT-IoT-v3.zip` | https://rdm.uq.edu.au/files/db0187cc-1f78-4215-8e34-303364498866 |
| D5 | LYCOS-IDS2017 | `data/raw/lycos/lycos-ids2017-master.zip` | https://maupiti-git.univ-lemans.fr/lycos/lycos-ids2017/archive/master.zip |

(UQ calls D1 "NF-CICIDS2018-v3"; save it under the name above.) Then register and check the files:

```bash
python scripts/download.py --register    # SHA-256, size, rows, columns -> data/MANIFEST.json
python scripts/d5_build_source.py        # runs the LYCOS authors' labelling script, writes the D5 source file
```

`data/MANIFEST.json` in this checkout records the SHA-256 of the files used for the paper; a different
hash means a different file version.

### 3. Build the leakage-controlled data packages

```bash
python scripts/prepare.py --resume       # D1-D5: audit, time blocks, windows, dedup, splits, LOACO folds,
                                         # cleaners, archives, D1-D3 -> other-dataset transfer packages
python scripts/p1c_build.py              # natural-novelty packages, D1 and D3
python scripts/p1d_build.py              # natural-novelty package, D5
python scripts/d5_purged_block.py        # purged-block LOACO split, D5
python scripts/rolling_build.py --build  # the 10 rolling cut-point packages (feasibility count is already in
                                         # results/final_v2/rolling_feasibility.json; do not re-run --count)
```

Every build runs fail-closed integrity checks and stops on the first failure.

### 4. Run the experiments and rebuild the reports

```bash
python scripts/final_run.py --resume     # 210 runs: 2 backbones, all detectors, 5 seeds -> results/final_v2/
python scripts/report_final.py           # results/final_v2/summary.json, RESULTS_SUMMARY.md, report/final/
python paper/noms2027/make_figures.py    # writes figures incl. the winners and operating-point figures
cp paper/noms2027/figures/fig_heatmap.pdf paper/hone/fig_winners.pdf
cp paper/noms2027/figures/fig_illusion.pdf paper/hone/fig_operating.pdf
python paper/hone/gen_appendix.py
python paper/hone/check_numbers.py
```

`--resume` skips runs whose result file already exists, so an interrupted run can be restarted. To
recompute every run, move `results/final_v2/` (except `rolling_feasibility.json`) out of the way first
and commit that change.

Results on other hardware can differ in the third decimal place (floating-point order on different CPUs
and library versions); the seeds, splits and code are identical.

## Where things are

| Path | Contents |
|---|---|
| `RESEARCH_QUESTIONS.md`, `docs/AMENDMENTS.json` | Pre-registration and the seven hashed amendments |
| `src/nids/` | Pipeline, models (AHSD, GRU), detectors, metrics |
| `scripts/` | Data builds, experiment runners, report generators |
| `results/final_v2/` | One JSON per evaluation x unit x seed, plus `summary.json` |
| `release/splits/` | The exact windows of every split (31 files) |
| `paper/hone/` | The HoNE paper (LaTeX), its figures and the number checks |
