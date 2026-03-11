"""
Clustering utilities: elbow method and silhouette analysis for best k.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def computeElbowScores(X_scaled, max_k=10):
    """Compute inertia for elbow method."""
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    return list(k_range), inertias


def computeSilhouetteScores(X_scaled, max_k=10):
    """Compute silhouette scores across k values."""
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    return list(k_range), silhouette_scores


def plotElbow(k_range, inertias, save_path):
    """Plot elbow curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, "bo-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved elbow plot: {save_path}")


def plotSilhouette(k_range, scores, save_path):
    """Plot silhouette scores."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, "ro-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Optimal k")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved silhouette plot: {save_path}")
    