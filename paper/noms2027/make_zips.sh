#!/usr/bin/env bash
# Build the two submission packages and test-compile the Overleaf one in a clean directory.
#   dist/noms2027_overleaf.zip : LaTeX sources only (main.tex, figures, tables) - upload to Overleaf as-is
#   dist/noms2027_full.zip     : sources + compiled PDF + the generators that rebuild every table/figure from JSON (results/final_v2)
set -euo pipefail
cd "$(dirname "$0")"
OUT=dist
rm -rf "$OUT" && mkdir -p "$OUT/overleaf/figures" "$OUT/overleaf/tables"
cp main.tex "$OUT/overleaf/"
for f in $(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' main.tex | sed -E 's/.*\{([^}]+)\}/\1/'); do
  cp "figures/$f.pdf" "$OUT/overleaf/figures/"
done
for t in $(grep -oE '\\input\{tables/[^}]+\}' main.tex | sed -E 's/.*tables\/([^}]+)\}/\1/'); do
  cp "tables/$t.tex" "$OUT/overleaf/tables/"
done
cat > "$OUT/overleaf/README.txt" <<'EOF'
IEEE/IFIP NOMS 2027 full paper (IEEEtran conference class, pdfLaTeX).
Compile: pdflatex main.tex (twice). Overleaf: upload this zip as a new project; main document = main.tex.
Before submission fill in: author block (must match EDAS exactly), [artifact URL], and an AI-use acknowledgment if applicable.
Tables in tables/ and figures in figures/ are generated from the experiment results JSON; do not edit by hand.
EOF
( cd "$OUT/overleaf" && zip -qr ../noms2027_overleaf.zip . )
# clean-room compile test
TMP=$(mktemp -d) && unzip -q "$OUT/noms2027_overleaf.zip" -d "$TMP"
( cd "$TMP" && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex > build.log )
grep -E "Overfull|undefined|Error" "$TMP/build.log" && { echo "clean-room compile has warnings"; exit 1; } || true
echo "clean-room compile OK: $(pdfinfo "$TMP/main.pdf" | grep Pages)"
# full package
mkdir -p "$OUT/full"
cp -r "$OUT/overleaf/." "$OUT/full/"
cp "$TMP/main.pdf" "$OUT/full/main.pdf"
mkdir -p "$OUT/full/generators"
cp make_tables.py make_figures.py check_numbers.py "$OUT/full/generators/"
cp ../../results/final_v2/summary.json "$OUT/full/generators/summary.json"
cat >> "$OUT/full/README.txt" <<'EOF'

generators/: make_tables.py, make_figures.py and check_numbers.py rebuild every table and figure and re-check
the numbers quoted in the text. They run inside the project repository (they read results/final_v2/summary.json,
the gate JSON of earlier phases and data/processed/*/index_meta.json); summary.json is included for reference.
EOF
( cd "$OUT/full" && zip -qr ../noms2027_full.zip . )
rm -rf "$TMP"
ls -la "$OUT"/*.zip
