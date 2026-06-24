# test_registry.py

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

print("Connecting to MLflow...")

model = mlflow.sklearn.load_model(
    "models:/TextClassifier@production"
)

print("Model loaded!")

result = model.predict(
    ["zomato order"]
)

print(result)