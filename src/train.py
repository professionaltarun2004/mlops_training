from src.data_loader import load_data
from src.preprocess import split_data 
import yaml 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle 
from src.logger import get_logger
from src.config_validator import validate_config
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import joblib
from sklearn.pipeline import Pipeline

logger = get_logger()

def train():
    try:
        logger.info("Loading config...")
        with open("config/config.yaml","r") as f:
            config=yaml.safe_load(f)

        validate_config(config)


        logger.info("Loading data...")
        data=load_data(config["data_path"])

        logger.info("Preprocessing data...")
        X_train,X_val,y_train,y_val=split_data(data)

        logger.info(f"Train labels distribution : {y_train.value_counts().to_dict()}")
        logger.info(f"Validation labels distribution: {y_val.value_counts().to_dict()}")

        logger.info("Starting mlflow run...")

        # logger.info("Vectorizing...")
        # vectorizer=CountVectorizer()
        # X_train_vec=vectorizer.fit_transform(X_train)
        # X_val_vec=vectorizer.transform(X_val)


        logger.info("Starting MLflow run...")

        mlflow.set_experiment("text_classifier_v3")

        with mlflow.start_run():
            mlflow.log_param("random_state",config["model_params"]["random_state"])
            mlflow.log_param("model_type","LogisticRegression") 
            mlflow.log_param("vectorizer", "CountVectorizer") 
            mlflow.log_param("data_version","20 samples")
            mlflow.log_param("C", config["model_params"]["C"])

            logger.info("Training model...")
            # model=LogisticRegression(
            #     random_state=config["model_params"]["random_state"]
            # )
            # model.fit(X_train_vec,y_train)
            pipeline = Pipeline([
                ('vectorizer',CountVectorizer()),
                ('model',LogisticRegression(
                    random_state=config["model_params"]["random_state"],
                    C=config["model_params"]["C"]
                ))
            ])
            pipeline.fit(X_train,y_train)

            train_preds=pipeline.predict(X_train) 
            val_preds=pipeline.predict(X_val)
            train_acc=accuracy_score(y_train,train_preds)
            val_acc=accuracy_score(y_val,val_preds)  

            mlflow.log_metric("train_accuracy",train_acc) 
            mlflow.log_metric("val_accuracy", val_acc)

            # mlflow.sklearn.log_model(model,name="model")

            # joblib.dump(vectorizer,"vectorizer.pkl")  
            # mlflow.log_artifact("vectorizer.pkl")

            mlflow.sklearn.log_model(pipeline,artifact_path="model")

        logger.info("Saving model...")
        joblib.dump(pipeline,config["model_path"])

        #with open(config["model_path"],"wb") as f:
            #pickle.dump((model,vectorizer),f)
        #print(f"Model saved successfully to {config['model_path']}")

        logger.info(f"Model saved successfully to {config['model_path']}")
        logger.info("Training completed succesfully")


        #pipeline automation
        import subprocess 

        result=subprocess.run(
            [
                "python",
                "-m",
                "src.select_best_model"
            ]
        )

        if result.returncode!=0:
            logger.error("Model selection pipeline failed")
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise e
            

if __name__=="__main__":
      train()