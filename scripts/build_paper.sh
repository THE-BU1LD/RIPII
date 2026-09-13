#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_TEMPLATE="$ROOT/paper/MANUSCRIPT.md"
OUTPUT="${1:-$ROOT/output/pdf/ripii-manuscript.pdf}"
PANDOC_BIN="${PANDOC_BIN:-$(command -v pandoc || true)}"
TECTONIC_BIN="${TECTONIC_BIN:-$(command -v tectonic || true)}"
PDFINFO_BIN="${PDFINFO_BIN:-$(command -v pdfinfo || true)}"
PDFTOTEXT_BIN="${PDFTOTEXT_BIN:-$(command -v pdftotext || true)}"
if [[ -z "$PDFINFO_BIN" && -x "/Users/ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo" ]]; then
  PDFINFO_BIN="/Users/ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo"
fi

test -s "$SOURCE_TEMPLATE"
for dependency in "$PANDOC_BIN" "$TECTONIC_BIN" "$PDFINFO_BIN"; do
  if [[ -z "$dependency" || ! -x "$dependency" ]]; then
    echo "paper build dependency is missing" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT")" "$ROOT/tmp/pdfs"
TEMP_PDF="$ROOT/tmp/pdfs/ripii-manuscript.pdf"
TEMP_TEXT="$ROOT/tmp/pdfs/ripii-manuscript.txt"
GENERATED_SOURCE="$ROOT/tmp/pdfs/MANUSCRIPT.generated.md"
rm -f "$TEMP_PDF" "$TEMP_TEXT" "$GENERATED_SOURCE"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

"${PYTHON_BIN:-python3}" "$ROOT/scripts/render_paper_results.py" \
  --template "$SOURCE_TEMPLATE" \
  --capsule "$ROOT/research/results/development/world_v3_convergence_capsule_v2.json" \
  --output "$GENERATED_SOURCE"

"$PANDOC_BIN" "$GENERATED_SOURCE" \
  --from=gfm \
  --standalone \
  --number-sections \
  --toc \
  --pdf-engine="$TECTONIC_BIN" \
  --variable=geometry:margin=1in \
  --variable=fontsize:10pt \
  --metadata=title:"RIPII Research Manuscript" \
  --output="$TEMP_PDF"

"$PDFINFO_BIN" "$TEMP_PDF" >/dev/null
if [[ -n "$PDFTOTEXT_BIN" && -x "$PDFTOTEXT_BIN" ]]; then
  "$PDFTOTEXT_BIN" "$TEMP_PDF" "$TEMP_TEXT"
  test -s "$TEMP_TEXT"
fi
mv "$TEMP_PDF" "$OUTPUT"
echo "built and text-verified $OUTPUT"
