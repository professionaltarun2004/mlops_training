import mlflow

#step 1 : set experiment 

experiment_name="text_classifier_v3"
mlflow.set_experiment(experiment_name)   

#step 2 : get experiment object
experiment=mlflow.get_experiment_by_name(experiment_name) 

#step 3 : fetch runs
runs=mlflow.search_runs(
    experiment_ids=[experiment.experiment_id]
) 

#step 4 : print runs
print(runs.columns)
print(runs[[
    "run_id","metrics.val_accuracy","params.C"
]])

#filter valid runs
runs=runs[runs["params.C"].notna()]

#ssafety check
if runs.empty:
    raise Exception("no valid runs found with proper logging")

best_run=runs.sort_values(
    by=["metrics.val_accuracy","start_time"],
    ascending=[False,False]
).iloc[0] 

best_run_id=best_run["run_id"]
best_accuracy=best_run["metrics.val_accuracy"]

print("\nbest run selected:")
print("run id", best_run_id)
print("validation accuracy:",best_accuracy)

#step 5: register best model
model_uri=f"runs:/{best_run_id}/model"

result=mlflow.register_model(
    model_uri=model_uri,
    name="TextClassifier"
)

print(f"\n registered model version: {result.version}")

from mlflow.tracking import MlflowClient 
client=MlflowClient()  

client.set_registered_model_alias(
    name="TextClassifier",
    version=result.version,
    alias="production"
)

print("Production alias updated!")