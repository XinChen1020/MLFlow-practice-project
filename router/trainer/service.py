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
        self._specs = specs or cfg.load_specs()
        self._roll = roll_service or RollService()
        self._ml = MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)

    # ---------- Public API ----------

    def train(
        self,
        trainer: str,
        *,
        wait_seconds: Optional[int] = None,
        image_key: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        spec = self._resolve_spec(trainer, image_key=image_key)

        name = self._unique_name(trainer)
        image = spec["trainer_image"]
        timeout = int(wait_seconds or spec.get("timeout") or cfg.TRAINER_TIMEOUT_SEC)
        gpus = spec.get("gpus")
        env = dict(spec.get("env") or {})
        env.setdefault("MLFLOW_TRACKING_URI", cfg.MLFLOW_TRACKING_URI)
        serve_image = spec.get("selected_serve_image")

        # Apply parameter overrides if there's any in the request
        applied_parameters: Dict[str, Any] = {}
        if parameters:
            for key, value in parameters.items():
                key_str = str(key)
                if value is None:
                    env.pop(key_str, None)
                    applied_parameters[key_str] = None
                    continue
                if isinstance(value, (str, bytes)):
                    env_value = value.decode("utf-8") if isinstance(value, bytes) else value
                elif isinstance(value, (int, float, bool)):
                    env_value = str(value)
                else:
                    env_value = json.dumps(value)
                env[key_str] = env_value
                applied_parameters[key_str] = env_value

        selected_image_key = spec.get("selected_image_key")
        image_key_for_log = selected_image_key if selected_image_key not in (None, "") else "default"

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

        print(
            "Starting trainer container "
            f"{name} "
            f"(image={image}, image_key={image_key_for_log}, serve_image={serve_image}, timeout={timeout}s, gpus={gpus}, "
            f"parameters={applied_parameters or None})"
        )

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
            "image_key": selected_image_key,
            "parameters": applied_parameters or None,
            "serve_image": serve_image
        }

        if status != 0 or version is None:
            raise HTTPException(
                status_code=500,
                detail=
                {
                    "error": f"trainer failed (exit={status})",
                    **resp,
                    "logs_tail": logs[-cfg.LOG_TAIL_ON_ERROR:],
                },
            )
            
        return resp

    def train_then_roll(
        self,
        trainer: str,
        *,
        wait_seconds: Optional[int] = None,
        image_key: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train and then roll out the produced model version using RollService.
        """
        train_resp = self.train(
            trainer,
            wait_seconds=wait_seconds,
            image_key=image_key,
            parameters=parameters,
        )
        print(f"Trainer completed: {train_resp}")
        version = train_resp.get("version")
        model_name = train_resp.get("registered_model")
        if not version or not model_name:
            raise HTTPException(status_code=500, detail="missing registered_model/version after training")
        try:
            roll_out = self._roll.roll(
                name=model_name,
                ref=version,
                wait_ready_seconds=cfg.HEALTH_TIMEOUT_SEC,
                serve_image=train_resp.get("serve_image"),
            )

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

    def _resolve_spec(self, trainer: str, image_key: Optional[str] = None) -> Dict[str, Any]:
        image_key = image_key or None
        if trainer not in self._specs:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "trainer spec not found",
                    "trainer": trainer,
                    "image_key": image_key,
                },
            )
        spec = dict(self._specs[trainer] or {})
        options_raw = spec.get("image_options") or {}
        if options_raw and not isinstance(options_raw, dict):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "image_options must be a mapping",
                    "trainer": trainer,
                    "image_key": image_key,
                },
            )
        options = dict(options_raw) if isinstance(options_raw, dict) else {}
        default_image = spec.get("trainer_image") or cfg.TRAINER_IMAGE
        selected_image = default_image
        if image_key:
            if image_key not in options:
                raise HTTPException(
                    status_code=400,
                    detail=
                    {
                        "error": "unknown trainer image key",
                        "trainer": trainer,
                        "image_key": image_key,
                        "available_keys": sorted(str(k) for k in options.keys()),
                    },
                )
            selected_image = options[image_key]
        if not selected_image:
            raise HTTPException(
                status_code=500,
                detail=
                {
                    "error": "trainer_image not configured",
                    "trainer": trainer,
                    "image_key": image_key,
                    "hint": (
                        "Set trainer_image in the trainer spec or define "
                        "the TRAINER_IMAGE environment variable."
                    ),
                },
            )
        serve_options_raw = spec.get("serve_image_options") or {}
        if serve_options_raw and not isinstance(serve_options_raw, dict):
            raise HTTPException(
                status_code=500,
                detail=
                {
                    "error": "serve_image_options must be a mapping",
                    "trainer": trainer,
                    "image_key": image_key,
                },
            )
        
        serve_options = dict(serve_options_raw) if isinstance(serve_options_raw, dict) else {}
        default_serve_image = spec.get("serve_image") or cfg.SERVE_IMAGE
        selected_serve_image = default_serve_image
        if image_key and image_key in serve_options:
            selected_serve_image = serve_options[image_key]

        spec["trainer_image"] = selected_image
        spec["image_options"] = options
        spec["selected_image_key"] = image_key
        spec["serve_image"] = default_serve_image
        spec["serve_image_options"] = serve_options
        spec["selected_serve_image"] = selected_serve_image

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
