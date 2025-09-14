from __future__ import annotations
import os, json, time, uuid
from typing import Dict, Any

import httpx
import docker
import mlflow
from mlflow.tracking import MlflowClient

import config as cfg

# ---- Clients ----
mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
ml_client = MlflowClient()
docker_client = docker.DockerClient(base_url=cfg.DOCKER_HOST)

# ---- State ----
DEFAULT_STATE = {"active": None, "url": None, "public_url": None, "model_uri": None, "ts": None}

def load_state() -> Dict[str, Any]:
    try:
        with open(cfg.STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return dict(DEFAULT_STATE)

def save_state(s: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cfg.STATE_PATH), exist_ok=True)
    tmp = cfg.STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, cfg.STATE_PATH)

# ---- Small shared utils ----

def unique(prefix: str) -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"


def ping(url: str, timeout: float = 3.0) -> bool:
    if not url:
        return False
    try:
        r = httpx.get(url.rstrip("/") + "/ping", timeout=timeout)
        return (200 <= r.status_code < 300) or (r.text.strip().upper() == "OK")
    except Exception:
        return False