from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import json, time, uuid

from fastapi import HTTPException
from docker.types import DeviceRequest

import config as cfg
from common import docker_client
from roll.service import RollService


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
        # (optional future fields: serve_image, params/presets, etc.)
      }
    }
    """

    def __init__(self, specs: Optional[Dict[str, Any]] = None, roll_service: Optional[RollService] = None):
        self._docker = docker_client
        self._specs = specs or cfg.load_trainer_specs()
        self._roll = roll_service or RollService()

    # ---------- Public API ----------

    def train(self, trainer: str, *, wait_seconds: Optional[int] = None) -> Dict[str, Any]:
        spec = self._resolve_spec(trainer)

        name    = self._unique_name(trainer)
        image   = spec["trainer_image"]
        timeout = int(wait_seconds or spec.get("timeout") or cfg.TRAINER_TIMEOUT_SEC)
        gpus    = spec.get("gpus")
        env     = dict(spec.get("env") or {})
        env.setdefault("MLFLOW_TRACKING_URI", cfg.MLFLOW_TRACKING_URI)
        print(f"Starting trainer container {name} (image={image}, timeout={timeout}s, gpus={gpus})")
        
        
        self._start_trainer(image=image, env=env, network=cfg.COMPOSE_NETWORK, name=name, gpus=gpus)
        status, logs = self._wait_trainer(name, timeout)

        result = self._parse_last_json_line(logs) or {}
        resp = {
            "container":        name,
            "run_id":           result.get("run_id"),
            "registered_model": result.get("registered_model"),
            "version":          result.get("version"),
            "alias_set":        result.get("alias"),
            "metrics":          result.get("metrics"),
        }

        if status != 0 and resp.get("version") is None:
            raise HTTPException(
                status_code=500,
                detail={"error": f"trainer failed (exit={status})", **resp, "logs_tail": logs[-cfg.LOG_TAIL_ON_ERROR:]},
            )
        return resp

    def train_then_roll(self, trainer: str, *, wait_seconds: Optional[int] = None) -> Dict[str, Any]:
        train_resp = self.train(trainer, wait_seconds=wait_seconds)
        version = train_resp.get("version")
        model_name = train_resp.get("registered_model")
        if not version or not model_name:
            raise HTTPException(status_code=500, detail="trainer did not emit registered_model/version in logs")
        try:
            roll_out = self._roll.roll(name=model_name, ref=version, wait_ready_seconds=cfg.HEALTH_TIMEOUT_SEC)
            return {**train_resp, "rolled": True, "public_url": roll_out.get("public_url")}
        except HTTPException as e:
            return {**train_resp, "rolled": False, "logs_tail": f"roll failed: {e.detail}"}

    # ---------- Helpers ----------

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
    def _parse_last_json_line(text: str) -> dict | None:
        for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
            try:
                return json.loads(line)
            except Exception:
                continue
        return None

    @staticmethod
    def _unique_name(trainer: str) -> str:
        return f"trainer-{trainer}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
