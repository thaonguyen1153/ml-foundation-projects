"""
CST2216 ML Projects - Interactive Model Predictor
Tabbed projects, shared prediction logic.
"""

import sys
import os
from pathlib import Path
import logging

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px
import base64
import json
from io import BytesIO
from typing import Tuple, Any

sys.path.append(".")
from src.config import APP_DIR, PROJECT_CONFIGS, MODELS_DIR, BASE_DIR, PROJECT_INPUTS, RANDOM_SEED, REPORT_DIR
from sklearn.preprocessing import MinMaxScaler
from src.models import createKMeans, fitModel
from src.evaluate import computeMetrics
from src.preprocessor import preprocessData, scaleTrainTest 
from src.utils import computeElbowScores, computeSilhouetteScores, plotElbow, plotSilhouette
from src.data_loader import loadData
    
logger = logging.getLogger(__name__)
logger.info(
    "Loading composite app from %s and models from %s",
    BASE_DIR,
    MODELS_DIR,
)

st.set_page_config(page_title="ML Predictor (Tabs)", layout="wide")

PROJECT_TABS = {
    "real_estate": {
        "emoji": "🏠",
        "title": "Real Estate Price Prediction",
        "img": "project_1.png",
    },
    "loan": {
        "emoji": "💳",
        "title": "Loan Approval Calculator",
        "img": "project_2.png",
    },
    "ucla": {
        "emoji": "🎓",
        "title": "UCLA Admission Predictor",
        "img": "project_4.png",
    },
    "clustering": {
        "emoji": "📊",
        "title": "Clustering Analysis",
        "img": "project_3.png",
    },
}


@st.cache_resource
def loadModelScaler(project, modelType):
    modelPath = Path(MODELS_DIR) / f"{project}_{modelType}_model.pkl"
    scalerPath = Path(MODELS_DIR) / f"{project}_{modelType}_scaler.pkl"

    try:
        model = joblib.load(str(modelPath))
        scaler = joblib.load(str(scalerPath))
        logger.info("Loaded model %s and scaler %s", modelPath, scalerPath)
        #st.success(f"✅ Loaded {modelPath.name}")
        return model, scaler
    except FileNotFoundError as exc:
        logger.exception("Model or scaler not found for %s (%s)", project, modelType)
        st.error(f"❌ Model or scaler not found: {exc}")
        st.stop()

def buildInputFrame(project):
    inputsConfig = PROJECT_INPUTS.get(project, {})
    featuresConfig = inputsConfig.get("features", {})

    userValues = {}
    inputFeatures = list(featuresConfig.keys())

    if project == "real_estate":
        numberFeatures = [(k, v) for k, v in featuresConfig.items() if v["type"] == "number"]
        cols = st.columns(3)
        for i, (feature, params) in enumerate(numberFeatures):
            col_idx = i % 3
            col = cols[col_idx]
            with col:
                label = feature.replace("_", " ").title()

                minVal = float(params["min"])
                maxVal = float(params["max"])
                defaultVal = float(params.get("value", minVal))
                stepVal = float(params.get("step", 1.0))

                if "value" not in params:
                    logger.warning(
                        "Missing 'value' for feature '%s' in project '%s', using minVal=%s",
                        feature, project, minVal,
                    )

                # Integer display for beds, baths, year_built, year_sold
                intFeatures = ["beds", "baths", "year_built", "year_sold"]
                if feature in intFeatures:
                    userValues[feature] = st.number_input(
                        label,
                        min_value=float(int(minVal)),  # float but whole number
                        max_value=float(int(maxVal)),
                        value=float(int(defaultVal)),
                        step=1.0,  # Float step of 1
                        key=f"{project}_{feature}",
                        format="%.0f",  # Displays as integer, no warning
                    )
                else:
                    userValues[feature] = st.number_input(
                        label,
                        min_value=minVal,
                        max_value=maxVal,
                        value=defaultVal,
                        step=stepVal,
                        key=f"{project}_{feature}",
                        format="%.2f",  # 2 decimals for others
                    )

        # Separate row for checkboxes (max 2)
        checkboxFeatures = [(k, v) for k, v in featuresConfig.items() if v["type"] == "checkbox"]
        if checkboxFeatures:
            checkboxCols = st.columns(min(2, len(checkboxFeatures)))
            for i, (feature, params) in enumerate(checkboxFeatures):
                with checkboxCols[i]:
                    label = feature.replace("_", " ").title()
                    checked = st.checkbox(
                        label,
                        value=bool(params.get("value", 0.0)),
                        key=f"{project}_{feature}",
                    )
                    userValues[feature] = 1.0 if checked else 0.0

        inputFrame = pd.DataFrame([userValues])

        # Computed features (unchanged)
        requiredFeatures = ["beds", "baths", "year_sold", "year_built"]
        missingFeatures = [feat for feat in requiredFeatures if feat not in inputFrame.columns]

        if missingFeatures:
            st.warning(f"⚠️ Missing real estate features: {missingFeatures}")
        else:
            bedsVal = inputFrame["beds"].iloc[0]
            bathsVal = inputFrame["baths"].iloc[0]
            inputFrame["popular"] = float(1 if (bedsVal == 2 and bathsVal == 2) else 0)
            yearSold = inputFrame["year_sold"].iloc[0]
            inputFrame["recession"] = float(1 if (2010 <= yearSold <= 2013) else 0)
            yearBuilt = inputFrame["year_built"].iloc[0]
            inputFrame["property_age"] = float(yearSold - yearBuilt)
            if inputFrame["property_age"].iloc[0] < 0:
                st.warning("⚠️ Invalid: property_age < 0. Setting to 0.")
                inputFrame["property_age"] = 0.0
    elif project == "loan":
        numberFeatures = [(k, v) for k, v in featuresConfig.items() if v["type"] == "number"]
        cols = st.columns(3)
        for i, (feature, params) in enumerate(numberFeatures):
            col_idx = i % 3
            col = cols[col_idx]
            with col:
                label = feature.replace("_", " ").title()

                minVal = float(params["min"])
                maxVal = float(params["max"])
                defaultVal = float(params.get("value", minVal))
                stepVal = float(params.get("step", 1.0))

                if "value" not in params:
                    logger.warning(
                        "Missing 'value' for feature '%s' in project '%s', using minVal=%s",
                        feature, project, minVal,
                    )

                # Integer display for beds, baths, year_built, year_sold
                intFeatures = ["dependents", "Loan_Amount_Term"]
                if feature in intFeatures:
                    userValues[feature] = st.number_input(
                        label,
                        min_value=float(int(minVal)),  # float but whole number
                        max_value=float(int(maxVal)),
                        value=float(int(defaultVal)),
                        step=1.0,  # Float step of 1
                        key=f"{project}_{feature}",
                        format="%.0f",  # Displays as integer, no warning
                    )
                else:
                    userValues[feature] = st.number_input(
                        label,
                        min_value=minVal,
                        max_value=maxVal,
                        value=defaultVal,
                        step=stepVal,
                        key=f"{project}_{feature}",
                        format="%.2f",  # 2 decimals for others
                    )

        # Separate row for checkboxes (max 2)
        checkbox_features = [(k, v) for k, v in featuresConfig.items() if v["type"] == "checkbox"]
        if checkbox_features:
            num_cols = min(2, len(checkbox_features))
            checkbox_cols = st.columns(num_cols)
            for i, (feature, params) in enumerate(checkbox_features):
                col_idx = i % num_cols  # SAFE: cycles 0,1,0,1...
                with checkbox_cols[col_idx]:
                    label = feature.replace("_", " ").title()
                    checked = st.checkbox(
                        label,
                        value=bool(params.get("value", False)),
                        key=f"{project}_{feature}",
                    )
                    userValues[feature] = 1.0 if checked else 0.0

        inputFrame = pd.DataFrame([userValues])

        # Computed features (unchanged)
        
    else:
        # Generic 3-column layout
        cols = st.columns(3)
        for i, (feature, params) in enumerate(featuresConfig.items()):
            col_idx = i % 3
            col = cols[col_idx]
            with col:
                label = feature.replace("_", " ").title()
                minVal = float(params["min"])
                maxVal = float(params["max"])
                defaultVal = float(params.get("value", minVal))
                stepVal = float(params.get("step", 1.0))

                if params["type"] == "number":
                    userValues[feature] = st.number_input(
                        label, min_value=minVal, max_value=maxVal,
                        value=defaultVal, step=stepVal,
                        key=f"{project}_{feature}",
                    )
                else:
                    userValues[feature] = st.slider(
                        label, min_value=minVal, max_value=maxVal,
                        value=defaultVal, key=f"{project}_{feature}",
                    )

        inputFrame = pd.DataFrame([userValues])

    return inputFrame

def prepare_loan_input(input_frame: pd.DataFrame, project: str) -> pd.DataFrame:
    """Transform Streamlit inputs to match trained model (20 features)."""
    if project != "loan":
        return input_frame
    
    # Safe value extraction helper
    def get_value(col: str, default: float | str = 0.0) -> float | str:
        if col in input_frame.columns:
            return input_frame[col].iloc[0]
        return default
    
    # Expected model columns
    model_cols = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History',
        'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes',
        'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
        'Education_Graduate', 'Education_Not Graduate',
        'Self_Employed_No', 'Self_Employed_Yes',
        'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
    ]
    
    model_input = pd.DataFrame(0.0, index=[0], columns=model_cols)
    
    # Numeric values
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
        model_input[col] = get_value(col, 0.0)
    
    # Gender
    male_val = get_value('male', 0.0)
    if male_val == 1.0:
        model_input['Gender_Male'] = 1.0
    else:
        model_input['Gender_Female'] = 1.0
    
    # Married
    married_val = get_value('maried', 0.0)
    if married_val == 1.0:
        model_input['Married_Yes'] = 1.0
    else:
        model_input['Married_No'] = 1.0
    
    # Dependents
    deps_val = int(get_value('dependents', 0))
    deps_map = {0: 'Dependents_0', 1: 'Dependents_1', 2: 'Dependents_2'}
    if deps_val in deps_map:
        model_input[deps_map[deps_val]] = 1.0
    else:
        model_input['Dependents_3+'] = 1.0
    
    # Education
    education_val = get_value('education', 'Graduate')
    if education_val == 'Graduate':
        model_input['Education_Graduate'] = 1.0
    else:
        model_input['Education_Not Graduate'] = 1.0
    
    # Self_Employed
    self_emp_val = get_value('self_employed', 0.0)
    if self_emp_val == 1.0:
        model_input['Self_Employed_Yes'] = 1.0
    else:
        model_input['Self_Employed_No'] = 1.0
    
    # Credit History
    model_input['Credit_History'] = get_value('Credit_History', 1.0)
    
    # Property Area
    area_val = get_value('property_area', 'Urban')
    area_map = {
        'Rural': 'Property_Area_Rural',
        'Semiurban': 'Property_Area_Semiurban', 
        'Urban': 'Property_Area_Urban'
    }
    if area_val in area_map:
        model_input[area_map[area_val]] = 1.0
    
    return model_input

def prepare_ucla_input(input_frame: pd.DataFrame, project: str) -> pd.DataFrame:
    """
    Transform Streamlit inputs to match UCLA trained model.
    Handles University_Rating (1-5) and Research (0/1) one-hot encoding.
    """
    if project != "ucla":
        return input_frame
    
    # Safe value extraction helper
    def get_value(col: str, default: float | str = 0.0) -> float | str:
        if col in input_frame.columns:
            return input_frame[col].iloc[0]
        return default
    
    # UCLA dummy columns (from preprocessing)
    dummy_cols = [
        'University_Rating_1', 'University_Rating_2', 'University_Rating_3', 
        'University_Rating_4', 'University_Rating_5',
        'Research_0', 'Research_1'
    ]
    
    # Start with zeros + copy numeric features
    model_cols = input_frame.columns.tolist() + dummy_cols  # Preserve originals + add dummies
    model_input = pd.DataFrame(0.0, index=[0], columns=model_cols)
    
    # Copy all existing numeric features
    for col in input_frame.columns:
        model_input[col] = get_value(col, 0.0)
    
    # University_Rating: map to one-hot (assume UI input 1-5)
    rating_val = int(get_value('university_rating', 1))  # Adjust UI column name
    if 1 <= rating_val <= 5:
        model_input[f'University_Rating_{rating_val}'] = 1.0
    else:
        model_input['University_Rating_1'] = 1.0  # Default
    
    # Research: map checkbox/slider to one-hot
    research_val = int(get_value('research', 0))
    if research_val == 1:
        model_input['Research_1'] = 1.0
    else:
        model_input['Research_0'] = 1.0
    
    # remove Research and University_Rating original columns if they exist, since model expects dummies
    model_input.drop(columns=['University_Rating', 'Research'], inplace=True, errors='ignore')
    
    logger.info("UCLA input prepared: %d features with dummies", len(model_input.columns))
    return model_input

def imageToBase64(pilImage):
    buffer = BytesIO()
    pilImage.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def render_model_selection(project: str) -> Tuple[str, Any, Any]:
    """Unified model selection with project-specific options and cache handling."""
    model_options = {
        "real_estate": ["linear", "random_forest"],
        "loan": ["logistic_regression", "decision_tree", "random_forest"]
    }
    
    options = model_options.get(project, ["random_forest"])
    st.subheader("Model Selection")
    
    model_type = st.selectbox(
        "Model Type",
        options,
        key=f"model_{project}"
    )
    
    # Cache management
    if "current_model" not in st.session_state:
        st.session_state.current_model = model_type
    
    if st.session_state.current_model != model_type:
        st.cache_resource.clear()
        st.session_state.current_model = model_type
        st.rerun()
    
    model, scaler = loadModelScaler(project, model_type)
    return model_type, model, scaler

def render_metrics(project: str, model_type: str) -> None:
    """Display project-specific metrics from JSON file."""
    metrics_path = MODELS_DIR / f"{project}_{model_type}_metrics.json"
    
    if not metrics_path.exists():
        st.info(f"No metrics at {metrics_path}")
        return
    
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)
    metrics = pd.DataFrame([metrics_dict])
    
    if project == "real_estate":
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("MAE", f"{metrics['mae'].iloc[0]:.2f}")
        with col2: st.metric("RMSE", f"{metrics['rmse'].iloc[0]:.2f}")
        with col3: st.metric("R²", f"{metrics['r2'].iloc[0]:.2f}")
    else:  # loan classification
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Accuracy", f"{metrics['accuracy'].iloc[0]:.2f}")
        with col2: st.metric("Precision", f"{metrics['precision'].iloc[0]:.2f}")
        with col3: st.metric("Recall", f"{metrics['recall'].iloc[0]:.2f}")
        with col4: st.metric("F1", f"{metrics['f1'].iloc[0]:.2f}")

@st.cache_data
def loadClusteringData():
    """Load raw clustering data."""
    csv_file = PROJECT_CONFIGS["clustering"]["raw"]
    df = loadData(csv_file, "clustering")
    return df

def performGenericClustering(df, selected_features, n_clusters):
    """Dynamic clustering pipeline using modular functions."""
    if len(selected_features) < 2:
        return None, None
    
    X = df[selected_features].copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reuse modular functions
    model = createKMeans(n_clusters=n_clusters)
    labels = fitModel(model, X_scaled)
    metrics = computeMetrics(X_scaled, labels)
    
    df_clustered = df.copy()
    df_clustered["cluster"] = labels
    
    return df_clustered, {
        "model": model,  # Added for centroids
        "n_clusters": len(set(labels)),
        "silhouette": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "scaler": scaler,
        "features": selected_features
    }
    
@st.cache_data
def analyzeBestK(X_scaled, max_k=10):
    """Analyze optimal k using elbow and silhouette."""
    k_range_elbow, inertias = computeElbowScores(X_scaled, max_k)
    k_range_sil, silhouette_scores = computeSilhouetteScores(X_scaled, max_k)
    
    # Suggest best k (highest silhouette)
    best_k = k_range_sil[np.argmax(silhouette_scores)]
    
    return {
        "k_range": k_range_sil,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "best_silhouette": max(silhouette_scores)
    }
def runProjectTab(project, tabData):
    emoji = tabData["emoji"]
    title = tabData["title"]
    imgName = tabData["img"]

    imgPath = Path(APP_DIR) / imgName
    logger.info("Trying to load image from %s", imgPath)

    if imgPath.is_file():
        try:
            image = Image.open(imgPath)
            #st.image(image, width="stretch")
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{imageToBase64(image)}" 
                        style="width: 100%; max-width: 100%; height: auto; border-radius: 8px;">
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            logger.exception("Failed to open image %s", imgPath)
            st.warning(f"⚠️ Could not open image {imgName}: {exc}")
    else:
        logger.warning("Image not found on disk: %s", imgPath)
        st.warning(f"❌ Image {imgName} not found at {imgPath}")


    # CLUSTERING has unique flow: feature selection + clustering visualization
    if project == "clustering":
        st.subheader("Clustering Analysis")
        
        df = loadClusteringData()
        st.write(f"Dataset: {len(df)} rows, {len(df.columns)} features")
        
        # Numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        #Find best k analysis
        # Best-k analysis button (before feature select)
        if st.button("Find best k clusters", type="secondary", key=f"best_k_{project}"):
            if len(numeric_features) >= 2:
                with st.spinner("Analyzing optimal k..."):
                    X_preview = MinMaxScaler().fit_transform(df[numeric_features[:3]])  # Use top 3
                    analysis = analyzeBestK(X_preview)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Suggested k", analysis["best_k"])
                        st.metric("Max Silhouette", f"{analysis['best_silhouette']:.3f}")
                    
                    # Elbow plot
                    plotElbow(analysis["k_range"], analysis["inertias"], os.path.join(REPORT_DIR, "imgs", "elbow.png"))
                    st.image("elbow.png", caption="Elbow Method")
                    
                    # Silhouette plot
                    plotSilhouette(analysis["k_range"], analysis["silhouette_scores"], os.path.join(REPORT_DIR, "imgs", "silhouette.png"))
                    st.image("silhouette.png", caption="Silhouette Scores")
                    
                    st.info(f"**Suggested: Use k={analysis['best_k']}**")
            else:
                st.warning("Need numeric features for analysis")
        
        # Feature selection + clustering parameters
        col1, col2 = st.columns(2)
        with col1:
            selected_features = st.multiselect(
                "Select features (min 2 for scatter)",
                options=numeric_features,
                default=numeric_features[:3] if len(numeric_features) >= 3 else numeric_features
            )
        with col2:
            n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
        
        if st.button("Run Clustering", type="primary", key=f"cluster_run_{project}") and len(selected_features) >= 2:
            with st.spinner("Clustering in progress..."):
                df_clustered, results = performGenericClustering(df, selected_features, n_clusters)
                model = results["model"]  # Now available
                n_clusters_actual = results["n_clusters"]
                
                if df_clustered is not None:
                    # Metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Clusters", results["n_clusters"])
                    with col_m2:
                        st.metric("Silhouette", f"{results['silhouette']:.3f}")
                    with col_m3:
                        st.metric("Calinski-Harabasz", f"{results['calinski_harabasz']:.0f}")
                    
                    # Interactive scatter plot with distinct colors
                    distinct_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Dark24

                    # Data points scatter
                    fig = px.scatter(
                        df_clustered,
                        x=selected_features[0],
                        y=selected_features[1],
                        color="cluster",
                        color_discrete_sequence=distinct_colors,
                        title=f"KMeans Clustering (k={n_clusters}) + Centroids",
                        hover_data=selected_features[2:5] if len(selected_features) > 2 else None,
                        opacity=0.85
                    )

                    # Centroid coordinates (back-transformed to original scale)
                    scaler = results["scaler"]
                    centroids_scaled = model.cluster_centers_
                    centroids_original = scaler.inverse_transform(centroids_scaled)

                    centroid_df = pd.DataFrame({
                        selected_features[0]: centroids_original[:, 0],
                        selected_features[1]: centroids_original[:, 1],
                        "cluster": range(n_clusters_actual)
                    })

                    # Add centroids as star markers
                    fig.add_scatter(
                        x=centroid_df[selected_features[0]],
                        y=centroid_df[selected_features[1]],
                        mode="markers",
                        marker=dict(
                            symbol="star",
                            size=18,
                            color=distinct_colors[:n_clusters_actual],
                            line=dict(width=2, color="black")
                        ),
                        name="Centroids",
                        text=[f"Centroid {i}" for i in range(n_clusters_actual)],
                        hovertemplate="<b>%{text}</b><br>" +
                                    f"{selected_features[0]}: %{{x:.3f}}<br>" +
                                    f"{selected_features[1]}: %{{y:.3f}}<extra></extra>",
                        showlegend=True
                    )

                    # Layout
                    fig.update_layout(
                        width=950,
                        height=700,
                        plot_bgcolor="white"
                    )
                    fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")), selector=dict(mode='markers'))
                    st.plotly_chart(fig, width='stretch')
                    # Centroid table (fixed - uses centroid_df columns)
                    st.subheader("Centroid Coordinates")
                    centroid_table = pd.DataFrame({
                        "Cluster": centroid_df["cluster"],
                        selected_features[0]: centroid_df[selected_features[0]].round(3),
                        selected_features[1]: centroid_df[selected_features[1]].round(3)
                    })
                    st.dataframe(centroid_table, width="stretch")
                    # Data preview
                    st.subheader("Clustered Data Preview")
                    st.dataframe(df_clustered[["cluster"] + selected_features].head(100))
                    
                    # Download
                    csv = df_clustered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download clustered data",
                        csv,
                        "clustered_data.csv",
                        "text/csv"
                    )
                else:
                    st.error("Need at least 2 features")
        else:
            st.info("👆 Select ≥2 features, set k, then click Run Clustering")
    else:
        # REAL ESTATE : Model type selection (regressor vs classifier)
        if project == "real_estate":
            st.subheader("Model Selection")
            modelType = st.selectbox(
                "Model Type",
                ["linear", "random_forest"],
                key=f"model_{project}",
            )
            
            # Force cache clear when model changes
            if "current_model" not in st.session_state:
                st.session_state.current_model = modelType
            
            if st.session_state.current_model != modelType:
                st.cache_resource.clear()  # Clear model cache
                st.session_state.current_model = modelType
                st.rerun()
            
            model, scaler = loadModelScaler(project, modelType)

            # Metrics (real_estate only)
            st.subheader("Training Metrics")
            metricsPath = MODELS_DIR / f"{project}_{modelType}_metrics.json"
            if metricsPath.exists():
                with open(metricsPath, "r") as f:
                    metricsDict = json.load(f)
                metrics = pd.DataFrame([metricsDict])
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("MAE", f"{metrics['mae'].iloc[0]:.2f}")
                with col2: st.metric("RMSE", f"{metrics['rmse'].iloc[0]:.2f}")
                with col3: st.metric("R²", f"{metrics['r2'].iloc[0]:.2f}")
            else:
                st.info(f"No metrics at {metricsPath}")
        elif project == "loan":
            st.subheader("Model Selection")
            modelType = st.selectbox(
                "Model Type",
                ["logistic", "decision_tree", "random_forest"],
                key=f"model_{project}",
            )
            
            model, scaler = loadModelScaler(project, modelType)

            # Metrics (real_estate only)
            st.subheader("Training Metrics")
            metricsPath = MODELS_DIR / f"{project}_{modelType}_metrics.json"
            if metricsPath.exists():
                with open(metricsPath, "r") as f:
                    metricsDict = json.load(f)
                metrics = pd.DataFrame([metricsDict])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Accuracy", f"{metrics['accuracy'].iloc[0]:.2f}")
                with col2: st.metric("Precision", f"{metrics['precision'].iloc[0]:.2f}")
                with col3: st.metric("Recall", f"{metrics['recall'].iloc[0]:.2f}")
                with col4: st.metric("F1", f"{metrics['f1'].iloc[0]:.2f}")
            else:
                st.info(f"No metrics at {metricsPath}")
        elif project == "ucla":     
            modelType="mlp"
            model, scaler = loadModelScaler(project, modelType)

            # Metrics (real_estate only)
            st.subheader("Training Metrics")
            metricsPath = MODELS_DIR / f"{project}_{modelType}_metrics.json"
            if metricsPath.exists():
                with open(metricsPath, "r") as f:
                    metricsDict = json.load(f)
                metrics = pd.DataFrame([metricsDict])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Accuracy", f"{metrics['accuracy'].iloc[0]:.2f}")
                with col2: st.metric("Precision", f"{metrics['precision'].iloc[0]:.2f}")
                with col3: st.metric("Recall", f"{metrics['recall'].iloc[0]:.2f}")
                with col4: st.metric("F1", f"{metrics['f1'].iloc[0]:.2f}")
            else:
                st.info(f"No metrics at {metricsPath}")
        # Common inputs for ALL projects
        st.subheader("Input Features")
        inputFrame = buildInputFrame(project)
        #st.write("DEBUG: Current inputFrame values:", inputFrame.iloc[0].to_dict()) 
        st.subheader("Input Preview")
        st.dataframe(inputFrame)

        # Predict ONLY for real_estate
        if project == "real_estate" and model is not None:
            if st.button("Predict", type="primary", key=f"predict_{project}"):
                XInput = inputFrame.values
                XScaled = scaler.transform(XInput)
                #st.write("DEBUG: Raw XInput shape:", XInput.shape)  
                #st.write("DEBUG: Sample scaled values:", XScaled[0][:3])  
                prediction = model.predict(XScaled)

                st.header("Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Price", f"${prediction[0]:,.0f}")
                with col2:
                    st.code(f"Raw Output: {prediction[0]}")
        elif project == "loan" and model is not None:
            if st.button("Predict", type="primary", key=f"predict_{project}"):
                # transform input to match model features (20 total with dummies)
                modelInput = prepare_loan_input(inputFrame, project)
                X_scaled = scaler.transform(modelInput)
                #st.write("DEBUG: Model input shape:", modelInput.shape)
                #st.write("DEBUG: Sample scaled columns:", modelInput.columns) 
                prediction = model.predict(X_scaled)

                st.header("Prediction Results")
                approvalStatus = "Approved" if prediction[0] == 1 else "Denied"
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Loan Status", approvalStatus)
                with col2:
                    st.code(f"Raw Output: {prediction[0]}")
        elif project == "ucla":
            if st.button("Predict", type="primary", key=f"predict_{project}"):
                modelInput = prepare_ucla_input(inputFrame, project)
                #st.write("DEBUG: Model input shape:", modelInput.shape)
                #st.write("DEBUG: Sample scaled columns:", modelInput.columns)
                #XInput = inputFrame.values
                X_scaled = scaler.transform(modelInput)
                prediction = model.predict(X_scaled)

                st.header("Prediction Results")
                approvalStatus = "High potential" if prediction[0] == 1 else "Low potential"
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Admit Chance", approvalStatus)
                with col2:
                    st.code(f"Raw Output: {prediction[0]}")
            
st.title("Algonquin College - BISI - Machine Learning 2 - ML Projects Dashboard")
st.markdown("*Personal Project • ThaoNguyen • 41168517*")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #FFD700 !important;
        border-radius: 8px 8px 0 0 !important;
        border-bottom: none !important;
        padding-top: 12px;
        padding-bottom: 8px;
        padding-left: 8px;
        padding-right: 8px;
        font-weight: 500;
        gap: 8px;
        transition: font-weight 0.2s ease !important;
    }
    
    /* Hover: Bold only, no color change */
    .stTabs [data-baseweb="tab"]:hover {
        font-weight: bold !important;
        background-color: #E8EBF0 !important;
    }
    
    /* Active tab: White bg + yellow bottom "line" like Windows folder */
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: black !important;
        font-weight: bold !important;
        box-shadow: 0 -4px 0 white !important;  /* Yellow underline */
    }
    
    /* Keep active tab text bold */
    .stTabs [aria-selected="true"] {
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)
tabNames = [f"{d['emoji']} {d['title']}" for d in PROJECT_TABS.values()]
tabs = st.tabs(tabNames)

for index, (projectKey, tabData) in enumerate(PROJECT_TABS.items()):
    with tabs[index]:
        runProjectTab(projectKey, tabData)

st.markdown("---")
st.markdown("*CST2216 ML Projects | Modular Pipeline + Streamlit*")
st.markdown(
    "Thanks for going through this small demo! "
    "If you have feedback, don't hesitate to reach out: "
    "**[nguy1153@algonquinlive.com](mailto:nguy1153@algonquinlive.com)**",
    unsafe_allow_html=True,
)
