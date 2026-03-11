import os
from pathlib import Path
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
APP_DIR = os.path.join(BASE_DIR, "streamlit")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
RANDOM_SEED = 42

LOG_DIR = os.path.join(REPORT_DIR, "logs")
LOG_DIR = Path(LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"ml_pipeline_{Path(BASE_DIR).name}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),  # Append to file
        logging.StreamHandler()  # Also print to console
    ]
)

PROJECT_CONFIGS = {
    "ucla": {
        "raw": "Admission.csv",
        "clean_file": "admission_cleaned.csv",
        "final_file": "admission_final.csv",
        "target_col": "Admit_Chance",
        "threshold": 0.8,      # target to have accuracy over 90%
        "drop_cols": ["Serial_No"],
        "stratify": True
    },
    "loan": {
        "raw": "credit.csv",
        "clean_file": "credit_cleaned.csv",
        "final_file": "credit_final.csv",
        "target_col": "Loan_Approved",  
        "threshold": None,
        "drop_cols": ["Loan_ID"],
        "stratify": True
    },
    "real_estate": {  # Regression
        "raw": "real_estate.csv",
        "clean_file": "real_estate_cleaned.csv",
        "final_file": "real_estate_final.csv",
        "feature_engineering": True,
        "target_col": "price",
        "threshold": None,
        "drop_cols": [],
        "stratify": False,
        "stratify_col": "property_type_Condo",
        "test_size": 0.2,
    },
    "clustering": {  # No target
        "raw": "mall_customers.csv",
        "target_col": None,
        "threshold": None,
        "drop_cols": [],
        "stratify": False
    }
}


PROJECT_INPUTS = {
    "real_estate": {
        "features": {
            "beds": {"type": "number", "min": 1.0, "max": 10.0, "value": 3.0, "step": 1.0},
            "baths": {"type": "number", "min": 1.0, "max": 8.0, "value": 2.0, "step": 0.5},
            "lot_size": {"type": "number", "min": 1000.0, "max": 500000.0, "value": 7500.0, "step": 500.0},
            "year_built": {"type":  "number", "min": 1900.0, "max": 2025.0, "value": 1995.0, "step": 1.0},
            "year_sold":  {"type":  "number", "min": 2000.0, "max": 2025.0, "value": 2020.0, "step": 1.0},
            "property_tax": {"type": "number", "min": 0.0, "max": 50000.0, "value": 5000.0, "step": 500.0},
            "insurance":    {"type": "number", "min": 0.0, "max": 20000.0, "value": 2000.0, "step": 250.0},
            "sqft":         {"type": "number", "min": 300.0, "max": 10000.0, "value": 1500.0, "step": 100.0},
            # Binary features handled via checkbox, but keep here for completeness
            "basement":           {"type": "checkbox", "value": 0.0},
            "property_type_Condo": {"type": "checkbox", "value": 0.0},
        }
    },
    "ucla": {
        "features": {
            "GRE_Score": {"type": "slider", "min": 200, "max": 340, "default": 310},
            "TOEFL_Score": {"type": "slider", "min": 80, "max": 120, "default": 105},
            "University_Rating": {"type": "slider", "min": 1, "max": 5, "default": 3},
            "SOP": {"type": "slider", "min": 1, "max": 5, "default": 3.5},
            "LOR": {"type": "slider", "min": 1, "max": 5, "default": 3.5},
            "CGPA": {"type": "slider", "min": 6, "max": 10, "default": 8.5},
            "Research": {"type": "slider", "min": 0, "max": 1, "default": 1}
        }
    },
    "loan": {
        "features": {
            "ApplicantIncome": {"type": "number", "min": 20000, "max": 500000, "default": 75000},
            "CoapplicantIncome": {"type": "number", "min": 0, "max": 500000, "default": 0},
            "LoanAmount": {"type": "number", "min": 5000, "max": 1000000, "default": 200000},
            "Loan_Amount_Term": {"type": "number", "min": 6, "max": 360, "default": 24, "step": 6},
            "dependents": {"type": "number", "min": 0, "max": 10, "default": 0, "step": 1},
            "education": {"type": "select", "options": ["Graduate", "Not Graduate"], "default": "Graduate"},
            "property_area": {"type": "select", "options": ["Urban", "Semiurban", "Rural"], "default": "Urban"},
            # Binary features handled via checkbox, but keep here for completeness
            "male":           {"type": "checkbox", "value": 0.0},
            "married": {"type": "checkbox", "value": 0.0},
            "self_employed": {"type": "checkbox", "value": 0.0},
            "Credit_History": {"type": "checkbox", "value": 0.0},
        }
    }
}

# Ensure directories exist
MODELS_DIR = Path(MODELS_DIR)
DATA_DIR = Path(DATA_DIR)
RAW_DIR = Path(RAW_DIR)
PROCESSED_DIR = Path(PROCESSED_DIR)

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)