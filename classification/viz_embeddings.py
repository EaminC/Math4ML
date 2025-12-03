#!/usr/bin/env python3
"""
Generate t-SNE scatter plots and k-means cluster visualizations for precomputed embeddings.

Example usage:
  python classification/viz_embeddings.py --dataset balanced --method word2vec --version original
  python classification/viz_embeddings.py --dataset balanced --method all --version all --sample-size 2000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1] / "Embedding"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "viz_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CHOICES = ["balanced", "unbalanced"]
METHOD_CHOICES = ["tfidf", "ngrams", "word2vec", "glove"]
VERSION_CHOICES = ["original", "pca"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize embeddings with t-SNE and k-means.")
    parser.add_argument(
        "--dataset",
        default="balanced",
        choices=DATASET_CHOICES + ["all"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--method",
        default="word2vec",
        choices=METHOD_CHOICES + ["all"],
        help="Embedding method.",
    )
    parser.add_argument(
        "--version",
        default="original",
        choices=VERSION_CHOICES + ["all"],
        help="Original or PCA embeddings.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of samples per class to visualize (after subsampling).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--kmeans-init",
        type=int,
        default=10,
        help="Number of initializations for k-means.",
    )
    return parser.parse_args()


def load_embeddings(dataset: str, method: str, version: str) -> Tuple[np.ndarray, np.ndarray]:
    base = BASE_DIR / dataset / method / version
    depressed = np.load(base / "depressed.npy")
    normal = np.load(base / "normal.npy")
    X = np.vstack([depressed, normal]).astype(np.float32, copy=False)
    y = np.concatenate(
        [np.ones(len(depressed), dtype=np.int8), np.zeros(len(normal), dtype=np.int8)]
    )
    return X, y


def subsample(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X_sub = []
    y_sub = []
    for label in [1, 0]:
        mask = np.where(y == label)[0]
        if len(mask) > sample_size:
            idx = rng.choice(mask, size=sample_size, replace=False)
        else:
            idx = mask
        X_sub.append(X[idx])
        y_sub.append(y[idx])
    X_concat = np.vstack(X_sub)
    y_concat = np.concatenate(y_sub)
    perm = rng.permutation(len(y_concat))
    return X_concat[perm], y_concat[perm]


def tsne_projection(
    X: np.ndarray,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    return tsne.fit_transform(X_scaled)


def plot_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    filename: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(
        coordinates[labels == 1, 0],
        coordinates[labels == 1, 1],
        c="#d62728",
        label="Depressed",
        s=15,
        alpha=0.7,
    )
    plt.scatter(
        coordinates[labels == 0, 0],
        coordinates[labels == 0, 1],
        c="#1f77b4",
        label="Normal",
        s=15,
        alpha=0.7,
    )
    plt.legend(frameon=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def run_kmeans_plot(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_init: int,
) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=2,
        random_state=random_state,
        n_init=n_init,
        max_iter=300,
    )
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    return cluster_labels


def plot_kmeans_confusion(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    title: str,
    filename: Path,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, cluster_labels)
    cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["cluster 0", "cluster 1"])
    ax.set_yticklabels(["normal", "depressed"])
    ax.set_xlabel("Cluster label")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    datasets = DATASET_CHOICES if args.dataset == "all" else [args.dataset]
    methods = METHOD_CHOICES if args.method == "all" else [args.method]
    versions = VERSION_CHOICES if args.version == "all" else [args.version]

    for dataset in datasets:
        for method in methods:
            for version in versions:
                print(f"Visualizing {dataset} / {method} / {version}")
                X, y = load_embeddings(dataset, method, version)
                X_sub, y_sub = subsample(
                    X, y, sample_size=args.sample_size, random_state=args.random_state
                )

                tsne_coords = tsne_projection(
                    X_sub,
                    perplexity=args.perplexity,
                    random_state=args.random_state,
                )
                tsne_file = (
                    OUTPUT_DIR
                    / f"{dataset}_{method}_{version}_tsne.png"
                )
                plot_scatter(
                    tsne_coords,
                    y_sub,
                    title=f"t-SNE: {dataset}/{method}/{version}",
                    filename=tsne_file,
                )

                clusters = run_kmeans_plot(
                    X_sub,
                    y_sub,
                    random_state=args.random_state,
                    n_init=args.kmeans_init,
                )
                kmeans_file = (
                    OUTPUT_DIR
                    / f"{dataset}_{method}_{version}_kmeans_confusion.png"
                )
                plot_kmeans_confusion(
                    y_sub,
                    clusters,
                    title=f"KMeans Confusion ({dataset}/{method}/{version})",
                    filename=kmeans_file,
                )

                print(f"  Saved t-SNE to {tsne_file}")
                print(f"  Saved KMeans confusion to {kmeans_file}")


if __name__ == "__main__":
    main()
