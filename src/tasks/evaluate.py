import pandas as pd
import os  
import yaml 
import json
from sklearn.metrics import accuracy_score
from src.logger import get_logger
import joblib

logger=get_logger()

def evaluate_task(val_path,model_path,metrics_path):
    try:
        model=joblib.open(model_path)
        data=pd.read_csv(val_path)
        X_val=data["text"]
        y_val=data["label"]

        preds=model.predict(X_val)

        accuracy=accuracy_score(y_val,preds)

        os.makedirs(os.path.dirname(metrics_path),exist_ok=True)
        metrics={
            "accuracy": accuracy
        }

        with open(metrics_path,"w") as f:
            json.dump(metrics,f)

        logger.info(f"evaluation complted. accuracy: {accuracy}")
        logger.info(f"metrics saved at {metrics_path}")

    except Exception as e:
        logger.error(f"error in evaluation: {e}")
        raise e 


