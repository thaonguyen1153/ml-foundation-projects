from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.config import RANDOM_SEED, PROCESSED_DIR
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def removeAttributes(df: pd.DataFrame, dropCols: List[str]) -> pd.DataFrame:
    """Remove specified columns (e.g., IDs)."""
    return df.drop(columns=dropCols, errors="ignore")

def binarizeTarget(df: pd.DataFrame, targetCol: str, threshold: Optional[float]) -> pd.DataFrame:
    """Binarize continuous target (e.g., AdmitChance > 0.8 → 1)."""
    if threshold is not None:
        df[targetCol] = (df[targetCol] > threshold).astype(int)
    return df

def categoricalEncoding(df: pd.DataFrame, targetCol: str) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = df.select_dtypes(include="object").columns.drop(targetCol, errors="ignore")
    return pd.get_dummies(df, columns=cat_cols, dtype=int)

def splitTrainTest(
    X: pd.DataFrame, 
    y: pd.Series, 
    testSize: float = 0.2, 
    randomState: int = 42,  # Default sklearn value
    stratifyCol: Optional[str] = None,  # ← NEW: Specific column
    stratifyTarget: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Notebook-exact split."""
    stratify_arg = None
    
    if stratifyCol and stratifyCol in X.columns:
        # Notebook style: stratify by specific feature column
        stratify_arg = X[stratifyCol]
        logger.info(f"Stratified split by: {stratifyCol}")
    elif stratifyTarget:
        # Generic: stratify by target
        stratify_arg = y
        logger.info("Stratified split by target")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=testSize, 
        random_state=randomState if randomState else None,
        stratify=stratify_arg
    )
    logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def scaleTrainTest(
    XTrain: pd.DataFrame, 
    XTest: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit scaler on train, transform both sets."""
    scaler = MinMaxScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)
    return XTrainScaled, XTestScaled, scaler

def preprocessData(df: pd.DataFrame, config: Dict[str, Any], project: str) -> Tuple:
    """Full pipeline: clean → encode → split → scale → save."""
    
    df = removeDuplicates(df)
    df = fillMissingBasement(df)
    df = filterLotSize(df)
    saveProcessed(df, project, config)  # Save cleaned CSV
    
    # Step 2: REAL ESTATE FEATURE ENGINEERING (explicit)
    if project == "real_estate" and config.get("feature_engineering"):
        logger.info("Applying Real Estate feature engineering...")
        df = realEstateFeatureEngineering(df)
        final_path = PROCESSED_DIR / config["final_file"]
        df.to_csv(final_path, index=False)
        logger.info(f"Saved final FE data: {final_path}")
        
    # Existing steps (drop, binarize, encode, split, scale)
    df = removeAttributes(df, config.get("drop_cols", []))
    df = binarizeTarget(df, config["target_col"], config.get("threshold"))
    df = categoricalEncoding(df, config["target_col"])
    
    # Step 4: Split
    target_col = config["target_col"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = splitTrainTest(
        X, y,
        testSize=config.get("test_size", 0.2),
        stratifyCol=config.get("stratify_col"),
        stratifyTarget=config.get("stratify_target", True)
    )
    
    # Step 5: Scale
    X_train_scaled, X_test_scaled, scaler = scaleTrainTest(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def preprocessLoanData(df: pd.DataFrame, config: Dict[str, Any], project: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, MinMaxScaler]:
    """
    Loan-specific preprocessing pipeline:
    - Impute mode/median for categoricals/numerics
    - Drop Loan_ID
    - Convert categoricals to object
    - One-hot encode categoricals
    - Encode target
    - Split and scale with MinMaxScaler
    """
    
    targetCol = config["target_col"]
    
    # Convert specific columns to object type
    objectCols = ['Credit_History', 'Loan_Amount_Term']
    for col in objectCols:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    # Impute categorical variables with mode
    categoricalCols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 
                      'Loan_Amount_Term', 'Credit_History']
    for col in categoricalCols:
        if col in df.columns:
            modeVal = df[col].mode()
            if len(modeVal) > 0:
                df[col] = df[col].fillna(modeVal[0])
            else:
                df[col] = df[col].fillna('Unknown')
    
    # Impute numerical variable with median
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    
    # Drop Loan_ID (redundant with config drop_cols)
    df = df.drop(columns=['Loan_ID'], errors='ignore')
    
    # One-hot encode categorical columns
    catCols = ['Gender', 'Married', 'Dependents', 'Education', 
               'Self_Employed', 'Property_Area']
    availableCats = [col for col in catCols if col in df.columns]
    df = pd.get_dummies(df, columns=availableCats, dtype=int)
    
    # Encode target: Y=1, N=0
    if targetCol in df.columns:
        df[targetCol] = df[targetCol].replace({'Y': 1, 'N': 0})
    
    # Save processed dataset
    finalPath = PROCESSED_DIR / config["final_file"]
    df.to_csv(finalPath, index=False)
    
    # Separate features and target
    X = df.drop(columns=[targetCol])
    y = df[targetCol]
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get("test_size", 0.2),
        stratify=y if config.get("stratify", True) else None,
        random_state=42
    )
    
    # Scale with MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Loan preprocessing complete. Train shape: {X_train_scaled.shape}")
    
    return pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index), \
           pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index), \
           y_train, y_test, scaler

def preprocessUCLAData(df: pd.DataFrame, config: Dict[str, Any], project: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, MinMaxScaler]:
    """
    UCLA-specific preprocessing:
    - Binarize target with threshold
    - Drop Serial_No
    - Convert categoricals to object + one-hot encode
    - Split with stratification
    - MinMax scale
    """
    target_col = config["target_col"]  # 'Admit_Chance'
    threshold = config.get("threshold", 0.8)
    
    # Step 1: Binarize target
    df[target_col] = (df[target_col] >= threshold).astype(int)
    
    # Step 2: Drop specified columns
    drop_cols = config.get("drop_cols", ["Serial_No"])
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Step 3: Convert to categorical + one-hot encode
    cat_cols = ['University_Rating', 'Research']
    available_cats = [col for col in cat_cols if col in df.columns]
    for col in available_cats:
        df[col] = df[col].astype('object')
    
    df = pd.get_dummies(df, columns=available_cats, dtype=int)
    
    # Save processed data
    final_path = PROCESSED_DIR / config["final_file"]
    df.to_csv(final_path, index=False)
    logger.info(f"Saved UCLA processed data: {final_path}")
    
    # Step 4: Split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get("test_size", 0.2),
        random_state=RANDOM_SEED,
        stratify=y if config.get("stratify", True) else None
    )
    
    # Step 5: MinMax scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"UCLA preprocessing complete. Train shape: {X_train_scaled.shape}")
    
    return (pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
            pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index),
            y_train, y_test, scaler)

def removeDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    initial_len = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_len - len(df)} duplicates")
    return df

def fillMissingBasement(df: pd.DataFrame, basement_col: str = "basement") -> pd.DataFrame:
    """NaN basement → 0 (no basement)."""
    initial_na = df[basement_col].isna().sum()
    df[basement_col] = df[basement_col].fillna(0)
    logger.info(f"Filled {initial_na} NaN basement → 0")
    return df

def filterLotSize(df: pd.DataFrame, lot_col: str = "lot_size", max_sqft: float = 500000) -> pd.DataFrame:
    """Remove lot_size > 500k sqft."""
    initial_len = len(df)
    df = df[df[lot_col] <= max_sqft]
    removed = initial_len - len(df)
    logger.info(f"Removed {removed} outliers (lot_size > {max_sqft:,} sqft)")
    return df

def saveProcessed(df: pd.DataFrame, project: str, config: Dict) -> str:
    """Save cleaned DF to data/processed/."""
    filename = config.get("clean_file", f"{project}_cleaned.csv")
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data: {path}")
    return str(path)

def realEstateFeatureEngineering(df: pd.DataFrame) -> pd.DataFrame:
    """Real Estate specific features."""
    # Popular: 2 beds + 2 baths
    df["popular"] = ((df["beds"] == 2) & (df["baths"] == 2)).astype(int)
    
    # Recession: 2010-2013
    df["recession"] = ((df["year_sold"] >= 2010) & (df["year_sold"] <= 2013)).astype(int)
    
    # Property age
    df["property_age"] = df["year_sold"] - df["year_built"]
    
    # Create dummy variables for 'property_type'
    df = pd.get_dummies(df, columns=['property_type'], drop_first=True).astype(int)

    # Filter invalid ages
    initial_len = len(df)
    df = df[df["property_age"] >= 0]
    logger.info(f"Feature engineering complete. Filtered {initial_len - len(df)} invalid ages.")
    
    return df