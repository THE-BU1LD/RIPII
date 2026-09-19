#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_markdown="$repo_root/paper/RIPII_MR_MANUSCRIPT.md"
output="${1:-$repo_root/output/pdf/ripii-mr-manuscript.pdf}"
pandoc_bin="${PANDOC_BIN:-$(command -v pandoc || true)}"
tectonic_bin="${TECTONIC_BIN:-$(command -v tectonic || true)}"
pdfinfo_bin="${PDFINFO_BIN:-$(command -v pdfinfo || true)}"
pdftotext_bin="${PDFTOTEXT_BIN:-$(command -v pdftotext || true)}"
if [[ -z "$pdfinfo_bin" && -x "/Users/ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo" ]]; then
  pdfinfo_bin="/Users/ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo"
fi

for dependency in "$pandoc_bin" "$tectonic_bin" "$pdfinfo_bin"; do
  if [[ -z "$dependency" || ! -x "$dependency" ]]; then
    echo "RIPII-MR paper build dependency is missing" >&2
    exit 1
  fi
done
test -s "$source_markdown"

mkdir -p "$(dirname "$output")" "$repo_root/tmp/pdfs"
temporary_pdf="$repo_root/tmp/pdfs/ripii-mr-manuscript.pdf"
temporary_text="$repo_root/tmp/pdfs/ripii-mr-manuscript.txt"
rm -f "$temporary_pdf" "$temporary_text"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

"$pandoc_bin" "$source_markdown" \
  --from=markdown+tex_math_single_backslash \
  --standalone \
  --number-sections \
  --pdf-engine="$tectonic_bin" \
  --variable=geometry:margin=1in \
  --variable=fontsize:10pt \
  --metadata=title:"RIPII-MR Prospective Research Manuscript" \
  --output="$temporary_pdf"

"$pdfinfo_bin" "$temporary_pdf" >/dev/null
if [[ -n "$pdftotext_bin" && -x "$pdftotext_bin" ]]; then
  "$pdftotext_bin" "$temporary_pdf" "$temporary_text"
  test -s "$temporary_text"
  rg -q "RIPII-MR Prospective Research Manuscript" "$temporary_text"
  rg -q "no accuracy, novelty" "$temporary_text"
  rg -q "RIPII-MR development matrix" "$temporary_text"
fi
mv "$temporary_pdf" "$output"
echo "built and text-verified $output"
