#!/usr/bin/env bash
# Run TF-IDF + NMF experiments and store metrics in dedicated JSON files.
#
# Usage:
#   bash classification/run_nmf.sh
#   NMF_COMPONENTS=80 DATASETS="balanced" EXTRA_ARGS="--cv-folds 3" bash classification/run_nmf.sh
#
# Environment overrides:
#   PYTHON_BIN      - Python interpreter (default: system Python 3.10)
#   DATASETS        - Space-separated list (default: "balanced unbalanced")
#   METHOD_ARG      - Embedding family (default: tfidf)
#   VERSION_ARG     - Embedding version (default: original)
#   NMF_COMPONENTS  - Number of NMF components (default: 64)
#   EXTRA_ARGS      - Additional CLI arguments for run_classification.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/classification/run_classification.py"
RESULT_DIR="$ROOT_DIR/classification/results"
mkdir -p "$RESULT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.10/bin/python3}"
DATASETS="${DATASETS:-balanced unbalanced}"
METHOD_ARG="${METHOD_ARG:-tfidf}"
VERSION_ARG="${VERSION_ARG:-original}"
NMF_COMPONENTS="${NMF_COMPONENTS:-64}"

EXTRA_ARGS_ARRAY=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
fi

run_for_dataset() {
  local dataset="$1"
  local output_file="$RESULT_DIR/${dataset}_${METHOD_ARG}_nmf${NMF_COMPONENTS}.json"
  local args=(
    --dataset "$dataset"
    --method "$METHOD_ARG"
    --version "$VERSION_ARG"
    --output "$output_file"
    --nmf-components "$NMF_COMPONENTS"
  )
  if [[ "$dataset" == "unbalanced" ]]; then
    args+=(--keep-class-imbalance)
  fi
  echo ">>> Running $dataset (results -> $output_file)"
  if [[ ${#EXTRA_ARGS_ARRAY[@]} -gt 0 ]]; then
    "$PYTHON_BIN" "$SCRIPT" "${args[@]}" "${EXTRA_ARGS_ARRAY[@]}"
  else
    "$PYTHON_BIN" "$SCRIPT" "${args[@]}"
  fi
  echo
}

for ds in $DATASETS; do
  run_for_dataset "$ds"
done

echo "All NMF runs completed. JSON metrics saved under $RESULT_DIR."
