# ml_flow.py
import mlflow
import mlflow.pytorch

def init_experiment(experiment_name="pytorch_image_classification"):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    return run

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict, step: int):
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)

def log_model(model, artifact_path="model"):
    mlflow.pytorch.log_model(model, artifact_path=artifact_path)

def end_run():
    mlflow.end_run()
