import mlflow

def register_model(run_id,model_name="TextClassifier"):

    model_uri=f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"Model registerd:  {result.name}, version: {result.version}")

if __name__=="__main__":
    run_id="686956c1043e4a17aaafdfb87a66a342"
    register_model(run_id)