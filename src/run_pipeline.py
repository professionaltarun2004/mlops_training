from src.tasks.load_data import load_data_task
from src.tasks.split_data import split_data_task
from src.tasks.train import train_task
from src.tasks.evaluate import evaluate_task
from src.logger import get_logger
import mlflow

logger=get_logger()

def run_pipeline():
    try:
        logger.info("starting pipeline...") 

        #add experiment
        mlflow.set_experiment("text_classifier_pipeline")

        #define paths

        raw_data_path="data/raw.csv"
        train_data_path="data/train.csv"
        val_data_path="data/val.csv"
        model_path="models/model.pkl"
        metrics_path="metrics/metrics.json"
        config_path="config/config.yaml"

        #mlflow runs from here
        with mlflow.start_run():

            load_data_task(
                input_path=config_path.replace("config/config.yaml", "data/sample.csv"),
                output_path=raw_data_path
            )

            split_data_task(
                input_path=raw_data_path,
                train_path=train_data_path,
                val_path=val_data_path
            )

            train_task(
                train_path=train_data_path,
                model_path=model_path,
                config_path=config_path
            )

            evaluate_task(
                val_path=val_data_path,
                model_path=model_path,
                metrics_path=metrics_path
            )

        logger.info("pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"pipeline failed : {e}")
        raise e
    
if __name__=="__main__":
    run_pipeline()
 