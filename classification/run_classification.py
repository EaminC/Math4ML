#!/usr/bin/env python3
"""
Utility to evaluate depression classifiers on pre-computed embeddings with an explicit
train/test split and an additional unsupervised baseline.

Usage examples:
  python classification/run_classification.py --dataset balanced --method tfidf --version original
  python classification/run_classification.py --dataset balanced --method all --version all --output results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

BASE_DIR = Path(__file__).resolve().parents[1] / "Embedding"
DATASET_CHOICES = ["balanced", "unbalanced"]
METHOD_CHOICES = ["tfidf", "ngrams", "word2vec", "glove"]
VERSION_CHOICES = ["original", "pca"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifiers on saved embeddings.")
    parser.add_argument(
        "--dataset",
        default="balanced",
        choices=DATASET_CHOICES,
        help="Which dataset split to use.",
    )
    parser.add_argument(
        "--method",
        default="tfidf",
        choices=METHOD_CHOICES + ["all"],
        help="Embedding family to evaluate, or 'all' for every method.",
    )
    parser.add_argument(
        "--version",
        default="pca",
        choices=VERSION_CHOICES + ["all"],
        help="Use PCA-compressed or original embeddings, or 'all'.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits used to create the hold-out test set.",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=0,
        help="Index of the fold (0-based) reserved as the test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling and model initialisation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file to store metrics.",
    )
    parser.set_defaults(balance_classes=None)
    parser.add_argument(
        "--balance-classes",
        dest="balance_classes",
        action="store_true",
        help="Force equal positive/negative counts via random downsampling.",
    )
    parser.add_argument(
        "--keep-class-imbalance",
        dest="balance_classes",
        action="store_false",
        help="Keep original class ratios.",
    )
    return parser.parse_args()


def load_embeddings(
    dataset: str,
    method: str,
    version: str,
    balance_classes: bool,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load numpy arrays and optionally align class sizes."""
    base = BASE_DIR / dataset / method / version
    depressed = np.load(base / "depressed.npy")
    normal = np.load(base / "normal.npy")

    rng = np.random.default_rng(random_state)

    if balance_classes:
        target_size = min(len(depressed), len(normal))
        if len(depressed) != target_size:
            idx = rng.choice(len(depressed), size=target_size, replace=False)
            depressed = depressed[idx]
        if len(normal) != target_size:
            idx = rng.choice(len(normal), size=target_size, replace=False)
            normal = normal[idx]

    X = np.vstack([depressed, normal]).astype(np.float32, copy=False)
    y = np.concatenate(
        [np.ones(len(depressed), dtype=np.int8), np.zeros(len(normal), dtype=np.int8)]
    )
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def build_supervised_models(random_state: int) -> Dict[str, Pipeline]:
    scaler = ("scaler", MaxAbsScaler())
    return {
        "log_reg": Pipeline(
            steps=[
                scaler,
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                scaler,
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        activation="relu",
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=100,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def select_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    test_fold: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if test_fold < 0 or test_fold >= n_splits:
        raise ValueError(f"test_fold must be between 0 and {n_splits - 1}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if fold_idx == test_fold:
            return train_idx, test_idx
    raise RuntimeError("Failed to obtain the requested test fold.")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
    }


def evaluate_supervised(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_scores)


def evaluate_kmeans(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> Dict[str, float]:
    kmeans = KMeans(
        n_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    kmeans.fit(X_train)

    train_clusters = kmeans.labels_
    fallback_class = int(np.argmax(np.bincount(y_train)))
    cluster_to_class: Dict[int, int] = {}
    for cluster_id in range(kmeans.n_clusters):
        indices = np.where(train_clusters == cluster_id)[0]
        if len(indices) == 0:
            cluster_to_class[cluster_id] = fallback_class
        else:
            counts = np.bincount(y_train[indices], minlength=2)
            cluster_to_class[cluster_id] = int(np.argmax(counts))

    test_clusters = kmeans.predict(X_test)
    y_pred = np.array([cluster_to_class[c] for c in test_clusters], dtype=np.int8)

    distances = kmeans.transform(X_test)
    weights = np.exp(-distances)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1.0
    weights /= weights_sum

    probs = np.zeros(len(y_test), dtype=np.float32)
    for cluster_id in range(kmeans.n_clusters):
        if cluster_to_class[cluster_id] == 1:
            probs += weights[:, cluster_id]

    return compute_metrics(y_test, y_pred, probs)


def main() -> None:
    args = parse_args()

    balance_classes = (
        args.balance_classes
        if args.balance_classes is not None
        else args.dataset == "balanced"
    )
    methods: List[str] = METHOD_CHOICES if args.method == "all" else [args.method]
    versions: List[str] = (
        VERSION_CHOICES if args.version == "all" else [args.version]
    )

    models = build_supervised_models(args.random_state)
    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for method in methods:
        aggregated.setdefault(method, {})
        for version in versions:
            print(
                f"\n=== Dataset: {args.dataset}, Method: {method}, Version: {version} ==="
            )
            X, y = load_embeddings(
                dataset=args.dataset,
                method=method,
                version=version,
                balance_classes=balance_classes,
                random_state=args.random_state,
            )
            print(
                f"Samples: {len(y)} (positive: {int(y.sum())}, negative: {len(y) - int(y.sum())})"
            )
            train_idx, test_idx = select_train_test_split(
                X,
                y,
                n_splits=args.cv_folds,
                test_fold=args.test_fold,
                random_state=args.random_state,
            )
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            aggregated[method].setdefault(version, {})

            for model_name, pipeline in models.items():
                metrics = evaluate_supervised(
                    pipeline, X_train, y_train, X_test, y_test
                )
                aggregated[method][version][model_name] = metrics
                print(
                    f"{model_name:>12s} | "
                    + " ".join(
                        f"{metric}:{metrics[metric]:.3f}" for metric in METRIC_NAMES
                    )
                )

            kmeans_metrics = evaluate_kmeans(
                X_train, y_train, X_test, y_test, random_state=args.random_state
            )
            aggregated[method][version]["kmeans_unsup"] = kmeans_metrics
            print(
                f"kmeans_unsup | "
                + " ".join(
                    f"{metric}:{kmeans_metrics[metric]:.3f}"
                    for metric in METRIC_NAMES
                )
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
