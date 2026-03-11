import pandas as pd
import logging
import os
from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def loadData(file_name: str, project: str) -> pd.DataFrame:
    """Load raw CSV into DataFrame."""
    #path = os.path.join(DATA_DIR, "raw", f"{project}.csv")
    path = os.path.join(DATA_DIR, "raw", file_name)
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File {path} not found")
        raise
