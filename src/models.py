"""ML Models for all 4 projects."""
# real estate models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

# Loan eligibility models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Admission models (NN)
from sklearn.neural_network import MLPClassifier

# Clustering models
from sklearn.cluster import KMeans, DBSCAN

import logging
from src.config import RANDOM_SEED


logger = logging.getLogger(__name__)
def trainLRRegressionModel(X_train, y_train):
    """Linear Regression."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression trained")
    return model

def trainRegressionModel(X_train, y_train):
    """Generic regression → LinearRegression."""
    return trainLRRegressionModel(X_train, y_train)

def trainRandomForestRegression(X_train, y_train):
    """Random Forest Regression."""
    #model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=123, n_jobs=-1)
    model = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    model.fit(X_train, y_train)
    logger.info("Random Forest Regression trained")
    return model

def trainLogisticRegressionClassifier(X_train, y_train):
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def trainDecisionTreeClassifier(X_train, y_train):
    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model

def trainRandomForestClassifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model

def trainMLPClassifier(X_train, y_train):
    """Train MLP Neural Network Classifier."""
    model = MLPClassifier(
        hidden_layer_sizes=(3),  # Neural net architecture
        batch_size=50,
        max_iter=200,
        random_state=RANDOM_SEED,
        #activation='tanh'
        #early_stopping=True
    )
    #MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=200, random_state=123)
    model.fit(X_train, y_train)
    return model

def createKMeans(n_clusters=3):
    """Create KMeans model with fixed parameters."""
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    logger.info(f"Created KMeans model with {n_clusters} clusters")
    return model


def createDbscan(eps=0.5, min_samples=5):
    """Create DBSCAN model with fixed parameters."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    logger.info(f"Created DBSCAN model (eps={eps}, min_samples={min_samples})")
    return model


def fitModel(model, X_scaled):
    """Fit model and return labels."""
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"Fitted model, found {n_clusters} clusters")
    return labels

def predictRegression(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred