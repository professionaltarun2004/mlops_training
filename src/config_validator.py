import os

def validate_config(config):
    if not os.path.exists(config["data_path"]):
        raise ValueError(f"Data path does not exist: {config['data_path']}")
    model_dir=os.path.dirname(config["model_path"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir,exist_ok=True)