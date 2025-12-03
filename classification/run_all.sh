#!/usr/bin/env bash
# Run every dataset/method/version combination using the system Python 3.10 interpreter.
#
# Usage (defaults run every dataset/method/version combination):
#   bash classification/run_all.sh
#   PYTHON_BIN=/path/to/python bash classification/run_all.sh
#
# Environment overrides:
#   DATASETS="balanced"            # choose datasets (space separated)
#   METHOD_ARG="tfidf"             # pass to --method (all|tfidf|ngrams|word2vec|glove)
#   VERSION_ARG="original"         # pass to --version (all|original|pca)
#   EXTRA_ARGS="--cv-folds 4 --test-fold 1"  # additional CLI flags for run_classification.py
#
# Outputs:
#   classification/results/balanced.json
#   classification/results/unbalanced.json

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/classification/run_classification.py"
RESULT_DIR="$ROOT_DIR/classification/results"
mkdir -p "$RESULT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.10/bin/python3}"
DATASETS="${DATASETS:-balanced unbalanced}"
METHOD_ARG="${METHOD_ARG:-all}"
VERSION_ARG="${VERSION_ARG:-all}"
NUM_FOLDS="${NUM_FOLDS:-5}"
TEST_FOLD="${TEST_FOLD:-0}"
EXTRA_ARGS_ARRAY=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
fi

run_for_dataset() {
  local dataset="$1"
  local output_file="$RESULT_DIR/${dataset}.json"
  local args=(
    --dataset "$dataset"
    --method "$METHOD_ARG"
    --version "$VERSION_ARG"
    --cv-folds "$NUM_FOLDS"
    --test-fold "$TEST_FOLD"
    --output "$output_file"
  )
  if [[ "$dataset" == "unbalanced" ]]; then
    args+=(--keep-class-imbalance)
  fi
  echo ">>> Running $dataset dataset (results -> $output_file)"
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

echo "All runs completed. JSON metrics saved under $RESULT_DIR."
