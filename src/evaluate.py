"""Evaluation metrics for all projects."""
# regression metrics:
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, 
                           classification_report, silhouette_score, mean_absolute_error)
# classification metrics:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#clustering 
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluateRegression(model, X_test, y_test):
    """Real Estate metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"Regression - MAE: {mae:.2f}, R²: {r2:.3f}, RMSE: {rmse:.2f}")
    return {"mae": mae, "r2": r2, "rmse": rmse}

def evaluateClassification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

def computeMetrics(X, labels):
    """Compute silhouette and Calinski-Harabasz scores."""
    n_unique = len(np.unique(labels))
    if n_unique > 1:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        metrics = {
            "silhouette": silhouette,
            "calinski_harabasz": calinski
        }
    else:
        metrics = {"silhouette": 0.0, "calinski_harabasz": 0.0}
    
    logger.info(f"Metrics - Silhouette: {metrics['silhouette']:.3f}, "
                f"CH: {metrics['calinski_harabasz']:.0f}")
    return metrics
