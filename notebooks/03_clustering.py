"""
Clustering Analysis Pipeline - KMeans & DBSCAN
No target variable, unsupervised learning
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import joblib
import json
import logging
import numpy as np

from src.models import createKMeans, createDbscan, fitModel
from src.evaluate import computeMetrics
from src.data_loader import loadData
from src.config import PROJECT_CONFIGS, MODELS_DIR, RANDOM_SEED

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def select_features(df, feature_cols):
    """Select specified features for clustering."""
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].copy()
    logger.info(f"Selected {len(available_features)} features: {available_features}")
    return X, available_features

def main():
    project = "clustering"
    config = PROJECT_CONFIGS[project]
    csv_file = config["raw"]
    
    parser = argparse.ArgumentParser(description="Clustering Pipeline")
    parser.add_argument("--method", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("--features", nargs="+", default=None)
    args = parser.parse_args()
    
    method = args.method
    feature_cols = args.features or config.get("features", [])
    
    try:
        logger.info(f"Loading {csv_file}...")
        df = loadData(csv_file, project)
        logger.info(f"Loaded {len(df)} rows")
        
        X, used_features = select_features(df, feature_cols)  # Use imported or define minimally
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == "kmeans":
            model = createKMeans(n_clusters=3)
        else:
            model = createDbscan()
        
        labels = fitModel(model, X_scaled)
        
        metrics = computeMetrics(X_scaled, labels)
        metrics["n_clusters"] = len(np.unique(labels))
        metrics["n_noise"] = list(labels).count(-1) if method == "dbscan" else 0
        
        # Save artifacts
        models_dir = MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        
        metrics_path = models_dir / f"{project}_{method}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")
        
        joblib.dump(model, models_dir / f"{project}_{method}_model.pkl")
        joblib.dump(scaler, models_dir / f"{project}_{method}_scaler.pkl")
        with open(models_dir / f"{project}_{method}_features.json", "w") as f:
            json.dump(used_features, f)
        
        logger.info(f"Saved artifacts for {project}_{method}")
        print(f"Clustering complete! Clusters: {metrics['n_clusters']}, "
              f"Silhouette: {metrics['silhouette']:.3f}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
