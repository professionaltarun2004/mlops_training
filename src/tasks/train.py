from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import yaml 
import os 
import joblib   
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import get_logger
import mlflow
import mlflow.sklearn

logger=get_logger()


def train_task(train_path,model_path,config_path):
    try:
        with open("config/config.yaml") as f:
            config=yaml.safe_load(f)
        
        data=pd.read_csv(train_path)
        X_train=data["text"]
        y_train=data["label"]

        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("model", LogisticRegression(
                    random_state=config["model_params"]["random_state"]
                ))
            ]
        )

        mlflow.log_param("random_state",config["model_params"]["random_state"])
        mlflow.log_param("model_type","LogisticRegression")
        mlflow.log_param("vectorizer","CountVectorizer")

        pipeline.fit(X_train,y_train)
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        joblib.dump(pipeline,model_path)

        mlflow.sklearn.log_model(pipeline,artifact_path="model")

        logger.info(f"model trained and saved at {model_path}")

    except Exception as e:
        logger.error(f"error in training {e}")
        raise e
