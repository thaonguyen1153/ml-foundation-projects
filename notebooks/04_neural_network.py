"""
UCLA Admission Prediction - Neural Network Pipeline
Binary Classification: MLPClassifier
"""

import sys
import os
import argparse
from pathlib import Path
import joblib
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import loadData
from src.preprocessor import preprocessUCLAData 
from src.models import trainMLPClassifier
from src.evaluate import evaluateClassification
from src.config import PROJECT_CONFIGS, RANDOM_SEED, MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # UCLA project config
    project = "ucla"
    config = PROJECT_CONFIGS[project]
    csv_file = config["raw"]
    
    parser = argparse.ArgumentParser(description="UCLA Admission Neural Network Pipeline")
    parser.add_argument('--model', choices=['mlp'], default='mlp', 
                        help="Model type: mlp (MLPClassifier)")
    args = parser.parse_args()
    
    model_type = 'mlp'  # Neural network
    
    try:
        logger.info(f"Loading {csv_file}...")
        df = loadData(csv_file, project)
        logger.info(f"Loaded {len(df)} rows")
        
        logger.info("Preprocessing...")
        X_train, X_test, y_train, y_test, scaler = preprocessUCLAData(df, config, project)
        
        logger.info("Training Neural Network...")
        model = trainMLPClassifier(X_train, y_train)
        
        logger.info("Evaluating...")
        metrics = evaluateClassification(model, X_test, y_test)
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        print(f"UCLA Admission MLP Accuracy: {metrics['accuracy']:.3f}")
        
        # Save metrics
        metrics_path = MODELS_DIR / f"{project}_{model_type}_metrics.json"
        metrics_path.parent.mkdir(exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save model + scaler
        model_path = MODELS_DIR / f"{project}_{model_type}_model.pkl"
        scaler_path = MODELS_DIR / f"{project}_{model_type}_scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved {model_path}, {scaler_path}")
        
        print("UCLA Neural Network Pipeline Complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
