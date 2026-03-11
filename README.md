# Algonquin College - CST2216 Individual Term Project

## Modularizing and Deploying ML Code

- Student: Thao Nguyen
- Course: CST2216 Business Intelligence System Infrastructure
- Due: Week 13

This repository modularizes 4 Jupyter Notebook projects from Level 1 (Real Estate, Loan Eligible, Clustering and Neural Network) into a production-ready VS Code framework with Streamlit deployment.

| Notebook                               | Project | ML Task                       |
| -------------------------------------- | ------- | ----------------------------- |
| Real_Estate.ipynb                      | Week 10 | Regression                    |
| Loan_Eligibility_Model_Solution.ipynb  | Week 11 | Classification                |
| Unsupervised_Clustering_Solution.ipynb | Week 12 | Clustering                    |
| UCLA_Neural_Networks_Solution.ipynb    | Week 13 | Neural Network Classification |

### 🚀 Quick Start
```
# Windows
git clone https://github.com/yourusername/ml-projects-framework.git
cd ml-projects-framework
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run any project
python .\notebooks\01_real_estate.py

# Streamlit Demo
streamlit run streamlit/ml-app.py
```

### 📁 Project Structure
```
ml-foundation-projects/
├── README.md                 # Setup, usage, pipeline overview
├── requirements.txt          # Dependencies (pandas, scikit-learn, etc.)
├── .gitignore                # Ignore __pycache__, .ipynb_checkpoints, data/
├── data/                     # Raw/external data (git-ignored)
│   ├── raw/                  # Original CSVs (e.g., loan.csv, Admission.csv)
│   ├── processed/            # Cleaned CSVs post-preprocessing
│   └── external/             # Public datasets if needed
├── notebooks/                # Converted/refactored notebooks - original notebooks
│   ├── 01_loan_eligibility.py
│   ├── 02_real_estate.py
│   ├── 03_ucla_nn.py
│   └── 04_clustering.py
├── src/                      # Reusable modules (core framework)
│   ├── __init__.py
│   ├── config.py             # Paths, seeds, thresholds (e.g., admit_chance=0.8)
│   ├── data_loader.py        # loadData(file_path) -> pd.DataFrame
│   ├── preprocessor.py       # cleanData(df, project='loan') -> X, y scaled
│   ├── models.py             # trainModel(X_train, y_train, model_type='nn')
│   ├── evaluate.py           # metrics(y_true, y_pred), confusion_matrix()
│   ├── pipeline.py           # runPipeline(project_name) -> model, metrics
│   └── utils.py              # plot, find best k...
├── models/                   # Saved models/artifacts (git-ignored) 
│   └── trained_models.pkl
├── reports/                  # Figures, logs (git-ignored)
│   ├── logs/
│   ├── imgs/
│   └── figures/
├── tests/                    # Unit tests for modules
│   └── test_preprocessor.py
└── streamlit/                # Future-ready (app.py for deployment)
    └── ml-app.py

```

### Key Features:

✅ Modular
✅ Logging
✅ Config-driven: Project-specific settings in config.py
✅ Clean Code
✅ Streamlit

### 🛠️ Modules Breakdown

| Module              | Purpose                     | Reusability  |
| ------------------- | --------------------------- | ------------ |
| src/config.py       | Paths, seeds, thresholds    | All projects |
| src/data_loader.py  | Load raw CSV → DataFrame    | All projects |
| src/preprocessor.py | Clean, scale, encode        | All projects |
| src/models.py       | Train (RF/NN/KMeans/etc.)   | All projects |
| src/pipeline.py     | End-to-end: load→train→save | All projects |
| src/evaluate.py     | Metrics, errors             | All projects |
| src/utils.py        | Utilities                   | All projects |

### Dataset

| Dataset                                | Project | ML Task                       |
| -------------------------------------- | --------------------------   | ----------------------------- |
| real_estate.csv                        | Real Estate price prediction | Regression                    |
| credit.csv  | Loan Acceptance prediction | Classification                |
| Admission.csv | Admission prediction | Neural Network Classification                    |
| mall_customers.csv    | Cluster data base on features | Clustering |

Preprocessing: Handle missing values, scale features, split 80/20 train/test.

### Methodology

| Models                                |  ML Task                       |
| --------------------------------------| ----------------------------- |
| linear regression, random forest regression                       | Regression                    |
| logistic regression, decision tree, random forest | Classification                |
| mlp | Neural Network Classification                    |
| kmeans | Clustering |

Evaluation: 
- Accuracy, Precision, Recall, F1-score for classification
- MAE, RSME, R2 for regression
- Silhouette Scores for clustering

### 🚀 Streamlit Deployment
Live Demo: [ml-foundation-projects](https://ml-foundation-projects.streamlit.app/)

Features:
- Input features for predictions
- Model selection dropdown
- Real-time metrics visualization
- Download predictions CSV
```
bash
streamlit run streamlit/ml-app.py --server.port 8501
# Deploy: Connect to GitHub → Streamlit Cloud
``` 
### 📈 Code Quality (Level 4 Rubric Compliance)
- **Modularization**: 7 reusable modules, single pipeline
- **Folder Structure**: Professional VS Code organization
- **Logging**: logging.INFO throughout with file rotation
- **Error Handling**: try/except + validation
- **Documentation**: Docstrings + this README
- **GitHub**: Public repo with complete history

### 📚 Dependencies
See requirements.txt.

### 🤝 GitHub Links:

[ML-foundation-projects](https://github.com/thaonguyen1153/ml-foundation-projects)

### 📄 Project Report
IEEE Format Report (6 pages: Intro, Methods, Results, Limitations)

### 🎤 Presentation
PowerPoint w/ Audio

### Limitations and Future Work
- Limitations: Small dataset; no real-time data support; assumes balanced classes.
- Future: Add models, fine-tune options, deploy to cloud, handle imbalanced data with SMOTE.

-----
Author:

Thao Nguyen - Contact: nguyenthi.ngocthao@gmail.com or nguy153@algonquinlive.com | CST2216 W26

Built with ❤️ for production ML deployment