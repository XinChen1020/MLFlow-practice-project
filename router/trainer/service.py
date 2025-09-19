from __future__ import annotations

import json
import time
import uuid
from typing import Dict, Any, Optional, Tuple

from fastapi import HTTPException
from docker.types import DeviceRequest

import config as cfg
from common import docker_client
from roll.service import RollService
from mlflow import MlflowClient


class TrainerService:
    """
    Spec-driven trainer orchestrator.

    Spec schema per trainer key (from YAML/JSON files):
    {
      "<trainer_key>": {
        "trainer_image": "image:tag",          # required
        "timeout": 1800,                       # optional (sec)
        "gpus": "all" | "1" | 1,               # optional
        "env": { ... }                         # optional; merged with MLFLOW_TRACKING_URI
      }
    }

    Flow:
      1) Resolve spec & environment.
      2) Pre-create an MLflow run and pass MLFLOW_RUN_ID to the trainer container.
      3) Start the trainer container and wait for completion.
      4) Query MLflow Model Registry for the version created by this run_id.
      5) (Optional) roll out via RollService.
    """

    def __init__(self, specs: Optional[Dict[str, Any]] = None, roll_service: Optional[RollService] = None):
        self._docker = docker_client
        self._specs = specs or cfg.load_trainer_specs()
        self._roll = roll_service or RollService()
        self._ml = MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)

    # ---------- Public API ----------

    def train(self, trainer: str, *, wait_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Launch a trainer container and return MLflow identifiers.

        Returns
        -------
        {
          "container": <name>,
          "run_id": <mlflow run id>,
          "registered_model": <model name>,
          "version": <model version>,
          "alias_set": null,
          "metrics": null
        }
        """
        spec = self._resolve_spec(trainer)

        name = self._unique_name(trainer)
        image = spec["trainer_image"]
        timeout = int(wait_seconds or spec.get("timeout") or cfg.TRAINER_TIMEOUT_SEC)
        gpus = spec.get("gpus")
        env = dict(spec.get("env") or {})
        env.setdefault("MLFLOW_TRACKING_URI", cfg.MLFLOW_TRACKING_URI)

        # Ensure we know the experiment & model name used by the trainer
        experiment = env.get("MLFLOW_EXPERIMENT", "diabetes_rf_demo")
        model_name = env.get("REGISTERED_MODEL_NAME", "DiabetesRF")

        # Pre-create an MLflow run and pass its ID to the container
        request_id = uuid.uuid4().hex
        run_id = self._create_run(experiment, {
            "request_id": request_id,
            "trainer": trainer,
            "container_name": name,
        })
        env["MLFLOW_RUN_ID"] = run_id

        print(f"Starting trainer container {name} (image={image}, timeout={timeout}s, gpus={gpus})")
        self._start_trainer(image=image, env=env, network=cfg.COMPOSE_NETWORK, name=name, gpus=gpus)
        status, logs = self._wait_trainer(name, timeout)

        version = None
        if status == 0:
            version = self._await_model_version(model_name, run_id)

        resp = {
            "container": name,
            "run_id": run_id,
            "registered_model": model_name,
            "version": version,
            "alias_set": None,
            "metrics": None,
        }

        if status != 0 or version is None:
            raise HTTPException(
                status_code=500,
                detail={"error": f"trainer failed (exit={status})", **resp, "logs_tail": logs[-cfg.LOG_TAIL_ON_ERROR:]},
            )
        return resp

    def train_then_roll(self, trainer: str, *, wait_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Train and then roll out the produced model version using RollService.
        """
        train_resp = self.train(trainer, wait_seconds=wait_seconds)
        print(f"Trainer completed: {train_resp}")
        version = train_resp.get("version")
        model_name = train_resp.get("registered_model")
        if not version or not model_name:
            raise HTTPException(status_code=500, detail="missing registered_model/version after training")
        try:
            roll_out = self._roll.roll(name=model_name, ref=version, wait_ready_seconds=cfg.HEALTH_TIMEOUT_SEC)
            return {**train_resp, "rolled": True, "public_url": roll_out.get("public_url")}
        except HTTPException as e:
            return {**train_resp, "rolled": False, "logs_tail": f"roll failed: {e.detail}"}

    # ---------- MLflow helpers ----------

    def _create_run(self, experiment: str, tags: Dict[str, str]) -> str:
        """
        Create an MLflow run in the given experiment and return run_id.
        Creates the experiment if it doesn't exist.
        """
        exp = self._ml.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else self._ml.create_experiment(experiment)
        run = self._ml.create_run(experiment_id=exp_id, tags=tags)
        return run.info.run_id

    def _await_model_version(self, name: str, run_id: str, tries: int = 30, sleep_s: float = 1.0) -> Optional[int]:
        """
        Poll the Model Registry for a version created by `run_id`.
        Returns the version number or None if not found within the time budget.
        """
        query = f"name = '{name}' and run_id = '{run_id}'"
        for _ in range(max(1, tries)):
            mvs = list(self._ml.search_model_versions(query))
            if mvs:
                try:
                    return int(mvs[0].version)
                except Exception:
                    pass
            time.sleep(sleep_s)
        return None

    # ---------- Docker helpers ----------

    def _resolve_spec(self, trainer: str) -> Dict[str, Any]:
        if trainer not in self._specs:
            raise HTTPException(status_code=404, detail=f"trainer spec not found: {trainer}")
        spec = dict(self._specs[trainer] or {})
        image = spec.get("trainer_image") or cfg.TRAINER_IMAGE
        spec["trainer_image"] = image
        # Normalize gpus field
        if "gpus" in spec and spec["gpus"] not in (None, ""):
            g = spec["gpus"]
            spec["gpus"] = "all" if str(g).lower() == "all" else int(g)
        return spec

    def _start_trainer(self, *, image: str, env: dict, network: str, name: str, gpus=None) -> str:
        kwargs = {
            "image": image,
            "name": name,
            "detach": True,
            "network": network,
            "environment": env,
            "labels": {"app": "trainer"},
        }
        if gpus:
            count = -1 if str(gpus).lower() == "all" else int(gpus)
            kwargs["device_requests"] = [DeviceRequest(count=count, capabilities=[["gpu"]])]
        self._docker.containers.run(**kwargs)
        return name

    def _wait_trainer(self, name: str, wait_seconds: int) -> Tuple[int, str]:
        """
        Wait for a container to finish and return (exit_code, logs_text).
        """
        status, logs_text = 1, ""
        try:
            c = self._docker.containers.get(name)
            try:
                status = int(c.wait(timeout=wait_seconds).get("StatusCode", 1))
            except Exception:
                status = 124  # timeout
            try:
                logs_text = c.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore")
            except Exception:
                pass
            try:
                c.remove(force=True)
            except Exception:
                pass
        except Exception:
            pass
        return status, logs_text

    @staticmethod
    def _unique_name(trainer: str) -> str:
        return f"trainer-{trainer}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
