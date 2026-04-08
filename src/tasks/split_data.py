
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.logger import get_logger
import yaml

with open("config/config.yaml") as f:
    config=yaml.safe_load(f)



logger=get_logger()

def split_data_task(input_path,train_path,val_path):

    import os  
    os.makedirs(os.path.dirname(train_path),exist_ok=True)
    os.makedirs(os.path.dirname(val_path),exist_ok=True)
    try:
        data=pd.read_csv(input_path)
        X=data["text"]
        y=data["label"]

        X_train,X_val,y_train,y_val=train_test_split(
            X,y, 
            test_size=0.2,
            stratify=y, 
            random_state=config["model_params"]["random_state"]
        )

        train_df=pd.DataFrame(
            {
                "text":X_train,
                "label":y_train
            }
        )
        train_df.to_csv(train_path,index=False)

        val_df=pd.DataFrame(
            {
                "text":X_val,
                "label":y_val
            }
        )
        val_df.to_csv(val_path,index=False)

        logger.info(f"train size: {len(train_df)}, val size:{len(val_df)}")
        logger.info(f"data split and saved to {train_path} and {val_path}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e