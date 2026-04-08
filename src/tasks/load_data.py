import pandas as pd 
from src.logger import get_logger

logger=get_logger()

def load_data_task(input_path,output_path):
    try:
        data=pd.read_csv(input_path)
        #save artifact
        data.to_csv(output_path,index=False)
        logger.info(f"data loaded from {input_path} and saved to {output_path}")

    except FileNotFoundError as e:
        logger.error(f"file not found {input_path}") 
        raise e 
    
    except Exception as e:
        logger.error(f"unexpected error: {e}") 
        raise e 