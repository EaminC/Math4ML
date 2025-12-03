#!/usr/bin/env bash
# Run advanced logistic regression variants (importance-weighted + LASSO) and
# store metrics in dedicated JSON files.
#
# Usage:
#   bash classification/run_advanced.sh
#   DATASETS="balanced" METHOD_ARG="tfidf" VERSION_ARG="original" bash classification/run_advanced.sh
#
# Environment overrides:
#   PYTHON_BIN   - Python interpreter (default: system python3.10)
#   DATASETS     - Space-separated datasets (default: "balanced unbalanced")
#   METHOD_ARG   - Embedding method (default: tfidf)
#   VERSION_ARG  - Embedding version (default: original)
#   MODES        - Space-separated modes to evaluate (default: "importance lasso")
#   EXTRA_ARGS   - Additional CLI options to forward to run_advanced_logistic.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/classification/run_advanced_logistic.py"
RESULT_DIR="$ROOT_DIR/classification/results"
mkdir -p "$RESULT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.10/bin/python3}"
DATASETS="${DATASETS:-balanced unbalanced}"
METHOD_ARG="${METHOD_ARG:-tfidf}"
VERSION_ARG="${VERSION_ARG:-original}"
MODES="${MODES:-importance lasso}"

EXTRA_ARGS_ARRAY=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
fi

for ds in $DATASETS; do
  output_file="$RESULT_DIR/advanced_${ds}_${METHOD_ARG}_${VERSION_ARG}.json"
  echo ">>> Running advanced modes (${MODES}) for dataset=$ds (results -> $output_file)"
  if [[ ${#EXTRA_ARGS_ARRAY[@]} -gt 0 ]]; then
    "$PYTHON_BIN" "$SCRIPT" \
      --dataset "$ds" \
      --method "$METHOD_ARG" \
      --version "$VERSION_ARG" \
      --modes $MODES \
      --output "$output_file" \
      "${EXTRA_ARGS_ARRAY[@]}"
  else
    "$PYTHON_BIN" "$SCRIPT" \
      --dataset "$ds" \
      --method "$METHOD_ARG" \
      --version "$VERSION_ARG" \
      --modes $MODES \
      --output "$output_file"
  fi
  echo
done

echo "Advanced logistic experiments completed."
