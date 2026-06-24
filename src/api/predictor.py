import mlflow.sklearn
from src.logger import get_logger
import csv
import os 
from datetime import datetime

logger=get_logger()

class Predictor: 

    def __init__(self):
        mlflow.set_tracking_uri(
            "http://host.docker.internal:5000"
        )
        self.model=mlflow.sklearn.load_model(
            "models:/TextClassifier@production"
        )
    
    def predict(self,text):
        prediction=self.model.predict([text])
        pred_value=prediction[0]

        log_file="logs/predictions.csv"

        file_exists=os.path.exists(log_file)

        with open(log_file, mode='a', newline='') as f:
            writer=csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ['timestamp','text','prediction']
                )
            
            current_time=datetime.now().strftime('%Y-%m-%d')
        
            writer.writerow([current_time,text,pred_value])
        logger.info(f"logged prediction for text: '{text}'")
        
        return pred_value
    