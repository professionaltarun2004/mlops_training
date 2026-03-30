import pandas as pd
from src.logger import get_logger

logger = get_logger()

def load_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {data_path}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error to load data from {data_path}: {e}")
        raise e