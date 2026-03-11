"""
Loan Eligibility Prediction - Full Pipeline
Classification: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
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

from src.pipeline import runPipeline  # noqa: F401
from src.data_loader import loadData
from src.preprocessor import preprocessLoanData
from src.models import (
    trainLogisticRegressionClassifier,
    trainDecisionTreeClassifier,
    trainRandomForestClassifier
)
from src.evaluate import evaluateClassification  # Use classification metrics
from src.config import PROJECT_CONFIGS, RANDOM_SEED, MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # Loan project config
    project = "loan"
    config = PROJECT_CONFIGS[project]
    csv_file = config["raw"]
    
    parser = argparse.ArgumentParser(description="Loan Eligibility Pipeline")
    parser.add_argument('--model', choices=['logistic', 'dt', 'rf'], default='logistic', 
                       help="Model type: logistic, dt (DecisionTree), rf (RandomForest)")
    args = parser.parse_args()
    
    model_type = {
        'logistic': 'logistic',
        'dt': 'decision_tree',
        'rf': 'random_forest'
    }.get(args.model, 'logistic')
    
    try:
        logger.info(f"Loading {csv_file}...")
        df = loadData(csv_file, project)
        logger.info(f"Loaded {len(df)} rows")
        
        logger.info("Preprocessing...")
        X_train, X_test, y_train, y_test, scaler = preprocessLoanData(df, config, project)
        
        logger.info("Training model...")
        if model_type == 'random_forest':
            model = trainRandomForestClassifier(X_train, y_train)
        elif model_type == 'decision_tree':
            model = trainDecisionTreeClassifier(X_train, y_train)
        else:  # logistic
            model = trainLogisticRegressionClassifier(X_train, y_train)
        
        logger.info("Evaluating...")
        metrics = evaluateClassification(model, X_test, y_test)  # Accuracy, Precision, Recall, F1
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        print(f"Loan Eligibility {model_type} Accuracy: {metrics['accuracy']:.3f}")
        
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
        
        print("Loan Eligibility Pipeline Complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
