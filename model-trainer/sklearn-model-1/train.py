#!/usr/bin/env python3
"""
RandomForestRegressor training with MLflow tracking and evaluation.

- Reads a CSV via DATASET_PATH + TARGET_COLUMN (or generates a diabetes demo CSV).
- Splits Train / Val / Test.
- Logs dataset lineage with mlflow.log_input (contexts: training, validation, evaluation).
- Logs parameters and registers the model.
- Uses mlflow.evaluate for canonical regression metrics/plots (no manual metrics).

Environment:
  MLFLOW_TRACKING_URI        MLflow tracking server URI
  MLFLOW_EXPERIMENT          Experiment name (default: "diabetes_rf_demo")
  REGISTERED_MODEL_NAME      Model Registry name (default: "DiabetesRF")
  DATASET_PATH               CSV path; if unset, a demo CSV is created under ./demo/data/diabetes.csv
  TARGET_COLUMN              Target column name (required if DATASET_PATH is set)
  TEST_SIZE                  Test fraction (default "0.2")
  VAL_SIZE                   Validation fraction of non-test portion (default "0.2")
  RANDOM_STATE               Random seed (default "42")
  N_ESTIMATORS               RF trees (default "200")
  MAX_DEPTH                  RF max depth (default "8")
  MLFLOW_RUN_ID              If set, reuse an existing run (from orchestrator); otherwise create a new run
  SET_ALIAS                  Optional alias to attach to the created model version (e.g., "staging")
  LOG_LEVEL                  Logging level (default "INFO")
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from helpers import prepare_demo_csv, load_csv, split_train_val_test, log_dataset_stage, resolve_version_for_run


# ---------- Logging & Experiment ----------
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "diabetes_rf_demo")
mlflow.set_experiment(EXPERIMENT)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("trainer")
client = MlflowClient()


# ---------- Main ----------

def main() -> None:
    # Data source selection
    dataset_path = os.getenv("DATASET_PATH")
    target_col = os.getenv("TARGET_COLUMN")
    if not dataset_path:
        dataset_path, target_col = prepare_demo_csv(os.getenv("DEMO_DIR", "./demo/data"))
        log.info("Using demo CSV: %s (target='%s')", dataset_path, target_col)
    elif not target_col:
        raise ValueError("TARGET_COLUMN must be set when DATASET_PATH is provided.")

    # Load & split
    X, y = load_csv(dataset_path, target_col)
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    val_size = float(os.getenv("VAL_SIZE", "0.2"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_val_test(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )

    # Hyperparameters
    n_estimators = int(os.getenv("N_ESTIMATORS", "200"))
    max_depth = int(os.getenv("MAX_DEPTH", "8"))
    registered_model = os.getenv("REGISTERED_MODEL_NAME", "DiabetesRF")

    run_id_env = os.getenv("MLFLOW_RUN_ID")  # set by orchestrator; optional

    # Use an existing run if provided; otherwise open a brand-new run
    with (mlflow.start_run(run_id=run_id_env) if run_id_env else mlflow.start_run()) as run:
        run_id = run.info.run_id
        mlflow.set_tags({"launch_mode": "orchestrated" if run_id_env else "standalone"})

        # Dataset lineage
        label = Path(dataset_path).resolve().as_uri()
        log_dataset_stage(X_tr, y_tr, stage="training",   name="train_dataset", source=label)
        log_dataset_stage(X_va, y_va, stage="validation", name="val_dataset",   source=label)
        log_dataset_stage(X_te, y_te, stage="evaluation", name="test_dataset",  source=label)

        # Parameters (data + model)
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "test_size": test_size,
                "val_size": val_size,
                "feature_count": X.shape[1],
                "sklearn_version": sklearn_version,
                "dataset_path": dataset_path,
                "target_column": target_col,
            }
        )

        # Train
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        # Log & register model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X.head(2),
            registered_model_name=registered_model,
        )


        # Standardized evaluation on test set (top-level API for broad compatibility)
        eval_df = X_te.copy()
        eval_df[target_col] = y_te.values
        result = mlflow.evaluate(
            model=f"runs:/{run_id}/model",
            data=eval_df,
            targets=target_col,
            model_type="regressor",
        )

        rmse = result.metrics.get("root_mean_squared_error")
        mae = result.metrics.get("mean_absolute_error")
        r2 = result.metrics.get("r2_score")
        log.info("Evaluation (test): RMSE=%.4f  MAE=%.4f  R2=%.4f", rmse, mae, r2)

        # Resolve model version for this run and optionally set alias
        version = resolve_version_for_run(client, registered_model, run_id)
        alias = os.getenv("SET_ALIAS")
        if alias and version is not None:
            client.set_registered_model_alias(registered_model, alias, int(version))
            log.info("Set alias '%s' on %s version %s", alias, registered_model, version)

        # Optional: emit a one-line summary for human/scripts (doesn't affect orchestrator path)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "registered_model": registered_model,
                    "version": version,
                    "alias": alias,
                    "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    import json  # local import to minimize global namespace
    main()
