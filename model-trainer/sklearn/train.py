import os
import json

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import numpy as np


# Set a clear experiment name (create if absent)
mlflow.set_experiment("diabetes_rf_demo")

client = MlflowClient()

def main():
    # Use pandas DataFrame so we have column names for the schema
    ds = load_diabetes(as_frame=True)
    X, y = ds.data, ds.target  # X: DataFrame with named columns; y: Series

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = int(os.getenv("N_ESTIMATORS", "200"))
    max_depth = int(os.getenv("MAX_DEPTH", "8"))
    random_state = 42

    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2  = float(r2_score(y_test, preds))

        # --- Params/Metrics ---
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        })
        
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        # --- Input example only
        input_example = X.head(2)

        # --- Log the model (and register if server supports registry) ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,          # triggers auto signature inference
            registered_model_name="DiabetesRF"    # works with SQL-backed tracking
        )

        print(f"Logged run. Metrics -> RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

        registered_model = "DiabetesRF"  # or read from env / arg
        # find the version that belongs to this run
        mv = next(iter(client.search_model_versions(
            f"name = '{registered_model}' and run_id = '{run.info.run_id}'"
        )), None)
        version = int(mv.version) if mv else None

        # if you want to set alias from env:
        alias = os.getenv("SET_ALIAS")
        if alias and version:
            client.set_registered_model_alias(registered_model, alias, version)

        # final structured line for the router to parse
        print(json.dumps({
            "run_id": run.info.run_id,
            "registered_model": registered_model,
            "version": version,
            "alias": alias,
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2}
        }))

if __name__ == "__main__":
    # If MLFLOW_TRACKING_URI isn't set, MLflow uses local ./mlruns
    print("MLFLOW_TRACKING_URI =", mlflow.get_tracking_uri())
    main()
