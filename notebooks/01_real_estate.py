"""
Real Estate Price Prediction - Full Pipeline
Runs end-to-end using shared src/ modules.
"""
import sys
import os
import argparse

from sklearn import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
import joblib
import json
from src.pipeline import runPipeline  # Orchestrates everything

# OR chain manually below:
from src.data_loader import loadData
from src.preprocessor import preprocessData
from src.models import trainLRRegressionModel, trainRandomForestRegression  # Define in models.py
from src.evaluate import evaluateRegression
from src.config import PROJECT_CONFIGS, RANDOM_SEED, MODELS_DIR

# Setup logging (rubric requirement)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Chain: Load → Preprocess → Train → Evaluate → Save."""
    
    project = "real_estate"
    config = PROJECT_CONFIGS[project]
    csv_file = config["raw"]
    
    parser = argparse.ArgumentParser(description="Real Estate Pipeline")
    parser.add_argument("--model", choices=["linear", "randomforest", "rf"], 
                       default="linear", help="Model type")
    args = parser.parse_args()
    
    model_type = {
        "linear": "linear",
        "randomforest": "random_forest",
        "rf": "random_forest"
    }.get(args.model, "linear")
    
    try:
        # 1. LOAD
        logger.info(f"Loading {csv_file}...")
        df = loadData(csv_file, project)
        logger.info(f"Loaded {len(df)} rows")
        
        # 2. PREPROCESS (chains remove, binarize, encode, split, scale)
        logger.info("Preprocessing...")
        X_train, X_test, y_train, y_test, scaler = preprocessData(df, config, project)
        
        # 3. TRAIN (implement in models.py)
        logger.info("Training model...")
        if model_type == "random_forest":
            model = trainRandomForestRegression(X_train, y_train)
        else:
            model = trainLRRegressionModel(X_train, y_train)  # e.g., RandomForestRegressor
        
        # 4. EVALUATE
        logger.info("Evaluating...")
        metrics = evaluateRegression(model, X_test, y_test)
        logger.info(f"MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.2f}")
        print(f"Real Estate R²: {metrics['r2']:.3f}")
        metricsPath = MODELS_DIR / f"{project}_{model_type}_metrics.json"
        metricsPath.parent.mkdir(exist_ok=True)
        with open(metricsPath, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metricsPath}")
        
        # 5. SAVE
        model_path = MODELS_DIR / f"{project}_{model_type}_model.pkl"
        scaler_path = MODELS_DIR / f"{project}_{model_type}_scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved model/scaler to {model_path}")
        
        print(f"Real Estate Pipeline Complete! MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.2f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
