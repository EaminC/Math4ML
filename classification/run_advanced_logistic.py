#!/usr/bin/env python3
"""
Evaluate advanced logistic regression variants:
  1) Importance-weighted logistic regression for covariate shift.
  2) Sparse (L1) logistic regression / LASSO.

Usage example:
  python classification/run_advanced_logistic.py --dataset balanced --method tfidf --version original \
      --modes importance lasso --output classification/results/advanced_balanced_tfidf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

BASE_DIR = Path(__file__).resolve().parents[1] / "Embedding"
DATASET_CHOICES = ["balanced", "unbalanced"]
METHOD_CHOICES = ["tfidf", "ngrams", "word2vec", "glove"]
VERSION_CHOICES = ["original", "pca"]
ALLOWED_MODES = ["importance", "lasso"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced logistic regression variants.")
    parser.add_argument(
        "--dataset",
        default="balanced",
        choices=DATASET_CHOICES,
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--method",
        default="tfidf",
        choices=METHOD_CHOICES,
        help="Embedding family.",
    )
    parser.add_argument(
        "--version",
        default="original",
        choices=VERSION_CHOICES,
        help="Embedding version (original or PCA).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["importance", "lasso"],
        choices=ALLOWED_MODES,
        help="Advanced variants to evaluate.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling and density ratio estimation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save aggregated JSON metrics.",
    )
    parser.add_argument(
        "--balance-classes",
        dest="balance_classes",
        action="store_true",
        help="Downsample majority class to match minority count.",
    )
    parser.add_argument(
        "--keep-class-imbalance",
        dest="balance_classes",
        action="store_false",
        help="Preserve natural class ratio.",
    )
    parser.set_defaults(balance_classes=None)
    return parser.parse_args()


def load_embeddings(
    dataset: str,
    method: str,
    version: str,
    balance_classes: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = BASE_DIR / dataset / method / version
    depressed = np.load(base / "depressed.npy")
    normal = np.load(base / "normal.npy")
    rng = np.random.default_rng(random_state)

    if balance_classes:
        target = min(len(depressed), len(normal))
        if len(depressed) != target:
            depressed = depressed[rng.choice(len(depressed), size=target, replace=False)]
        if len(normal) != target:
            normal = normal[rng.choice(len(normal), size=target, replace=False)]

    X = np.vstack([depressed, normal]).astype(np.float32, copy=False)
    y = np.concatenate(
        [np.ones(len(depressed), dtype=np.int8), np.zeros(len(normal), dtype=np.int8)]
    )
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def compute_density_ratios(
    X: np.ndarray,
    random_state: int,
    test_fraction: float = 0.3,
) -> np.ndarray:
    """Estimate density ratio r(x) = p_test(x) / p_train(x) via logistic calibration."""
    rng = np.random.default_rng(random_state)
    domain_labels = (rng.random(len(X)) < test_fraction).astype(np.int8)
    # Ensure both domains are present
    if domain_labels.mean() == 0:
        domain_labels[rng.integers(len(X))] = 1
    elif domain_labels.mean() == 1:
        domain_labels[rng.integers(len(X))] = 0

    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    density_model = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )
    density_model.fit(X_scaled, domain_labels)
    probs = density_model.predict_proba(X_scaled)
    eps = 1e-6
    ratios = probs[:, 1] / np.clip(probs[:, 0], eps, None)
    return ratios.astype(np.float32)


def summarize(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {
        "mean": {name: float(np.mean(values)) for name, values in metrics.items()},
        "std": {name: float(np.std(values)) for name, values in metrics.items()},
    }


def evaluate_mode(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    importance_weights: np.ndarray | None,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    metrics = {name: [] for name in METRIC_NAMES}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        pipeline_steps = [("scaler", MaxAbsScaler())]
        if mode == "importance":
            clf = LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
            fit_kwargs = {"clf__sample_weight": importance_weights[train_idx]}
        else:  # lasso
            clf = LogisticRegression(
                solver="saga",
                penalty="l1",
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )
            fit_kwargs = {}

        pipeline_steps.append(("clf", clf))
        pipeline = Pipeline(pipeline_steps)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train, **fit_kwargs)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))

        print(
            f"  Fold {fold_idx}: "
            + " ".join(f"{name}={metrics[name][-1]:.3f}" for name in METRIC_NAMES)
        )

    return summarize(metrics)


def main() -> None:
    args = parse_args()
    balance_classes = (
        args.balance_classes if args.balance_classes is not None else args.dataset == "balanced"
    )
    X, y = load_embeddings(
        dataset=args.dataset,
        method=args.method,
        version=args.version,
        balance_classes=balance_classes,
        random_state=args.random_state,
    )
    print(
        f"Loaded dataset={args.dataset}, method={args.method}, version={args.version} "
        f"with {len(y)} samples (positives={int(y.sum())}, negatives={len(y) - int(y.sum())})"
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    importance_weights = None
    if "importance" in args.modes:
        print("Computing density ratio weights for importance-weighted logistic regression...")
        importance_weights = compute_density_ratios(X, random_state=args.random_state)

    for mode in args.modes:
        print(f"\n=== Evaluating mode: {mode} ===")
        mode_result = evaluate_mode(
            mode=mode,
            X=X,
            y=y,
            cv=cv,
            importance_weights=importance_weights,
            random_state=args.random_state,
        )
        results[mode] = mode_result
        print(
            f"{mode} summary: "
            + " ".join(f"{metric}={mode_result['mean'][metric]:.3f}" for metric in METRIC_NAMES)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved advanced metrics to {args.output}")


if __name__ == "__main__":
    main()
