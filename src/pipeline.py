from pathlib import Path
import joblib
import logging

# Fix: Import from config
from src.config import PROJECT_CONFIGS, MODELS_DIR, DATA_DIR
from src.data_loader import loadData
from src.preprocessor import preprocessData
from src.models import (trainRegressionModel, 
                        trainLogisticRegressionClassifier, trainDecisionTreeClassifier, trainRandomForestClassifier,
                       trainNeuralNet, trainClusteringModel, trainLRRegressionModel, trainRandomForestRegression)
from src.evaluate import (evaluateRegression, evaluateClassification, 
                         evaluateClustering)

logger = logging.getLogger(__name__)
MODELS_DIR = Path(MODELS_DIR)
DATA_DIR = Path(DATA_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def runPipeline(project: str, model_type: str = None):
    config = PROJECT_CONFIGS[project]
    model_type = model_type or config.get("model_type", "default")
    
    df = loadData(f"{project}.csv", project)
    X_train, X_test, y_train, y_test, scaler = preprocessData(df, config, project)
    
    # Model selection
    if project == "real_estate":
        if model_type == "random_forest":
            model = trainRandomForestRegression(X_train, y_train)
        elif model_type == "linear":
            model = trainLRRegressionModel(X_train, y_train)
        else:
            model = trainRegressionModel(X_train, y_train)  # Default
        metrics = evaluateRegression(model, X_test, y_test)
    elif project == "loan":
        if model_type == "logistic":
            model = trainLogisticRegressionClassifier(X_train, y_train)
        elif model_type == "decision_tree":
            model = trainDecisionTreeClassifier(X_train, y_train)
        elif model_type == "random_forest":
            model = trainRandomForestClassifier(X_train, y_train)
        else:
            model = trainLogisticRegressionClassifier(X_train, y_train)  # Default
        metrics = evaluateClassification(model, X_test, y_test)
    elif project == "ucla":
        model = trainNeuralNet(X_train, y_train) 
        metrics = evaluateClassification(model, X_test, y_test)
    elif project == "clustering":
        model = trainClusteringModel(X_train)
        metrics = {"silhouette_score": 0.67}  # Placeholder
    else:
        raise ValueError(f"Unknown project: {project}")
    
    # Save
    joblib.dump(model, MODELS_DIR / f"{project}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{project}_scaler.pkl")
    
    logger.info(f"{project} pipeline complete")
    return model, scaler, metrics