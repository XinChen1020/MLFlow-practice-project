"""
helpers.py — small, documented utilities for data IO, splitting, and dataset lineage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def prepare_demo_csv(dst_dir: str = "./demo/data") -> Tuple[str, str]:
    """
    Create a Diabetes CSV (regression) at `dst_dir` and return (path, target_col).

    Parameters
    ----------
    dst_dir : str
        Destination directory for the CSV file.

    Returns
    -------
    path : str
        Filesystem path to the created CSV.
    target_col : str
        Name of the target column in the CSV ("target").
    """
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    df = load_diabetes(as_frame=True).frame
    path = Path(dst_dir) / "diabetes.csv"
    df.to_csv(path, index=False)
    return str(path), "target"


def load_csv(path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV file into numeric features `X` and target `y`.

    Notes
    -----
    - Non-numeric columns are dropped for simplicity (RandomForestRegressor demo).
    - Upstream preprocessing/pipelines can be added later if needed.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    target_col : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix.
    y : pd.Series
        Target vector.

    Raises
    ------
    ValueError
        If the target column is missing.
    """
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"TARGET_COLUMN '{target_col}' not found in: {path}")

    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split into Train / Validation / Test.

    The validation split is taken from the non-test remainder so that final
    proportions reflect `test_size` and `val_size`.

    Parameters
    ----------
    X, y : pd.DataFrame, pd.Series
        Full dataset.
    test_size : float
        Fraction for the test set (0 < test_size < 1).
    val_size : float
        Fraction of the non-test portion for validation (0 < val_size < 1).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_rel = val_size / max(1e-9, (1.0 - test_size))
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def log_dataset_stage(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    stage: str,
    source: str,
    name: str,
) -> None:
    """
    Log dataset lineage for a pipeline stage using MLflow's Dataset API.

    This attaches a named dataset to the active run with a `context`
    (e.g., "training", "validation", "evaluation") and logs a few
    lightweight data-quality hints.

    Parameters
    ----------
    X, y : pd.DataFrame, pd.Series
        Features and target for this stage.
    stage : str
        Context name shown in MLflow ("training" | "validation" | "evaluation").
    source : str
        Human-readable URI (e.g., "file:///.../dataset.csv") shown in MLflow.
    name : str
        Display name of the dataset in MLflow.
    """
    df = X.copy()
    df["target"] = y.values if hasattr(y, "values") else y
    ds = mlflow.data.from_pandas(df, source=source, name=name)
    mlflow.log_input(ds, context=stage)

    # Small quality indicators for quick sanity checks in the UI
    missing_pct = (df.isnull().to_numpy().sum() / float(df.size)) * 100.0
    mlflow.log_metrics(
        {
            f"{stage}_rows": len(df),
            f"{stage}_columns": len(df.columns),
            f"{stage}_missing_pct": float(missing_pct),
        }
    )
def resolve_version_for_run(
    client,
    model_name: str,
    run_id: str,
    tries: int = 20,
    sleep_s: float = 0.5,
) -> Optional[int]:
    """
    Look up the Model Registry version created by a specific MLflow run.
    Returns the version number or None if not found within the retry window.
    """
    query = f"name = '{model_name}' and run_id = '{run_id}'"
    for _ in range(max(1, tries)):
        mvs = list(client.search_model_versions(query))
        if mvs:
            try:
                return int(mvs[0].version)
            except Exception:
                pass
        time.sleep(sleep_s)
    return None