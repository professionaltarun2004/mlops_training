from sklearn.model_selection import train_test_split
from src.logger import get_logger

logger = get_logger()

def split_data(data):
    try:
        X=data["text"]
        y=data["label"]

        X_train,X_val,y_train,y_val=train_test_split(
            X,y,test_size=0.2,stratify=y, random_state=42
        )

        logger.info("Data split into train and validation")

        return X_train,X_val,y_train,y_val
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e
    
    