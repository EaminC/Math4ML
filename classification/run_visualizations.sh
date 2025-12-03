#!/usr/bin/env bash
# Generate t-SNE and k-means plots for the requested datasets/methods/versions.
#
# Usage:
#   bash classification/run_visualizations.sh
#   DATASETS="balanced" METHODS="word2vec glove" VERSIONS="original" bash classification/run_visualizations.sh
#
# Environment overrides:
#   PYTHON_BIN    - Python interpreter (default: system python3.10)
#   DATASETS      - Space-separated list (default: "balanced unbalanced")
#   METHODS       - Space-separated methods (default: "tfidf ngrams word2vec glove")
#   VERSIONS      - Space-separated versions (default: "original pca")
#   SAMPLE_SIZE   - Samples per class (default: 500)
#   PERPLEXITY    - t-SNE perplexity (default: 35)
#   RANDOM_STATE  - Seed (default: 42)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/classification/viz_embeddings.py"
RESULT_DIR="$ROOT_DIR/viz_outputs"
mkdir -p "$RESULT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.10/bin/python3}"
DATASETS="${DATASETS:-balanced unbalanced}"
METHODS="${METHODS:-tfidf ngrams word2vec glove}"
VERSIONS="${VERSIONS:-original pca}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
PERPLEXITY="${PERPLEXITY:-35}"
RANDOM_STATE="${RANDOM_STATE:-42}"

for dataset in $DATASETS; do
  for method in $METHODS; do
    for version in $VERSIONS; do
      echo ">>> Visualizing dataset=$dataset method=$method version=$version"
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}" \
      "$PYTHON_BIN" "$SCRIPT" \
        --dataset "$dataset" \
        --method "$method" \
        --version "$version" \
        --sample-size "$SAMPLE_SIZE" \
        --perplexity "$PERPLEXITY" \
        --random-state "$RANDOM_STATE"
      echo
    done
  done
done

echo "Visualization complete. Files saved under $RESULT_DIR."
