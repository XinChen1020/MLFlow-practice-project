# router/main.py
"""
FastAPI router for blue/green ML deployments (fixed public port via Caddy proxy)
-------------------------------------------------------------------------------
- Generic runtime + SERVE_MODEL_URI (serve containers have NO host port)
- Caddy listens on a fixed host port (e.g., 9000) and reverse-proxies to the active container
- Roll flow: start candidate -> health check -> flip proxy -> drain -> retire old

Endpoints:
- GET  /status
- POST /admin/train
- POST /admin/roll
- POST /admin/train_then_roll
"""

import os, time, json, uuid
from typing import Optional, Union, Dict, Any, Tuple

import httpx
import docker
import mlflow
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

# ---------- Env ----------
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DOCKER_HOST          = os.getenv("DOCKER_HOST", "http://socket-proxy:2375")

SERVE_IMAGE          = os.getenv("SERVE_IMAGE", "mlserve-runtime:sklearn")  # your generic runtime
SERVE_PORT           = int(os.getenv("SERVE_PORT", "8080"))                 # container listen port

STATE_PATH           = os.getenv("STATE_PATH", "/state/active.json")
COMPOSE_NETWORK      = os.getenv("COMPOSE_NETWORK", f"{os.getenv('COMPOSE_PROJECT_NAME', 'local')}_default")
HEALTH_TIMEOUT_SEC   = int(os.getenv("HEALTH_TIMEOUT_SEC", "90"))
DRAIN_GRACE_SEC      = float(os.getenv("DRAIN_GRACE_SEC", "1.5"))           # wait after proxy flip

TRAINER_IMAGE        = os.getenv("TRAINER_IMAGE", "model-trainer:latest")
TRAINER_TIMEOUT_SEC  = int(os.getenv("TRAINER_TIMEOUT_SEC", "1800"))        # 30m default
LOG_TAIL_ON_ERROR    = int(os.getenv("TRAINER_LOG_TAIL", "1200"))

# Caddy reverse proxy (data plane) & public address
PROXY_PUBLIC_PORT    = int(os.getenv("PROXY_PUBLIC_PORT", "9000"))          # Caddy's listener
PROXY_ADMIN_URL      = os.getenv("PROXY_ADMIN_URL", "http://proxy:2019")    # Caddy Admin API (inside network)
PUBLIC_HOST          = os.getenv("PUBLIC_HOST", "localhost")                # shown in status/public_url

# ---------- Clients ----------
docker_client = docker.DockerClient(base_url=DOCKER_HOST)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
ml_client = MlflowClient()

# ---------- Schemas ----------
class StatusResp(BaseModel):
    active: Optional[str] = None
    url: Optional[str] = None           # router-to-container internal URL
    public_url: Optional[str] = None    # fixed proxy URL (host-facing)
    healthy: bool = False
    ts: Optional[float] = None

class RollReq(BaseModel):
    name: str = Field(..., examples=["DiabetesRF"])
    ref: Union[str, int] = Field(..., examples=["@production", 17])  # alias or version
    wait_ready_seconds: int = Field(default=HEALTH_TIMEOUT_SEC, ge=1, le=900)

class RollResp(BaseModel):
    active: str
    url: str            # internal
    public_url: str     # fixed proxy URL

class TrainReq(BaseModel):
    experiment: str = Field(default="diabetes_rf_demo")
    register_model_name: str = Field(default="DiabetesRF")
    set_alias: str | None = Field(default=None)  # e.g., "staging" or "production"
    n_estimators: conint(ge=1, le=5000) = 200
    max_depth:   conint(ge=1, le=512)   = 8
    wait_seconds: conint(ge=1, le=24*3600) = TRAINER_TIMEOUT_SEC

class TrainResp(BaseModel):
    container: str
    run_id: str | None = None
    registered_model: str | None = None
    version: int | None = None
    alias_set: str | None = None
    metrics: dict | None = None
    logs_tail: str | None = None

class TrainRollResp(TrainResp):
    rolled: bool = False

# ---------- State helpers ----------
def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"active": None, "url": None, "public_url": None, "model_uri": None, "ts": None}

def _save_state(s: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)

# ---------- Utils ----------
def _unique(prefix: str) -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"

def _ping(url: str, timeout: float = 3.0) -> bool:
    try:
        r = httpx.get(url.rstrip("/") + "/ping", timeout=timeout)
        return (200 <= r.status_code < 300) or (r.text.strip().upper() == "OK")
    except Exception:
        return False

def _resolve_models_uri(name: str, ref: Union[str, int]) -> str:
    try:
        if isinstance(ref, str) and ref.startswith("@"):
            mv = ml_client.get_model_version_by_alias(name, ref[1:])
            return f"models:/{name}/{mv.version}"
        if isinstance(ref, int):
            return f"models:/{name}/{ref}"
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"model/alias not found: {e}")
    raise HTTPException(status_code=400, detail="ref must be '@alias' or integer version")

def _start_generic_runtime(name: str, network: str, model_uri: str) -> str:
    """
    Start a serving container on the compose network (NO host port publishing).
    Returns container id.
    """
    c = docker_client.containers.run(
        image=SERVE_IMAGE,
        name=name,
        detach=True,
        network=network,
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "SERVE_MODEL_URI": model_uri,
            "SERVE_PORT": str(SERVE_PORT),
        },
        restart_policy={"Name": "on-failure", "MaximumRetryCount": 1},
        labels={"app": "mlflow-serve"},
    )
    return c.id

def _retire(name: Optional[str]) -> None:
    if not name: return
    try:
        c = docker_client.containers.get(name)
        c.stop(timeout=5); c.remove()
    except Exception:
        pass

def _make_caddy_config(target_container: str) -> Dict[str, Any]:
    """
    Build a minimal Caddy config that reverse-proxies :9000 -> target_container:SERVE_PORT.
    """
    return {
        "admin": { "listen": ":2019" },
        "apps": {
            "http": {
                "servers": {
                    "srv0": {
                        "listen": [ f":{PROXY_PUBLIC_PORT}" ],
                        "routes": [{
                            "handle": [{
                                "handler": "reverse_proxy",
                                "upstreams": [{ "dial": f"{target_container}:{SERVE_PORT}" }]
                            }]
                        }]
                    }
                }
            }
        }
    }

def _proxy_point_to(target_container: str) -> None:
    """
    Atomically replace Caddy config so all NEW requests go to target_container.
    Existing in-flight requests keep flowing to the previously active container.
    """
    cfg = _make_caddy_config(target_container)
    r = httpx.put(PROXY_ADMIN_URL.rstrip("/") + "/load", json=cfg, timeout=5.0)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"proxy switch failed: {e.response.text}")

def _start_trainer(req: TrainReq, network: str) -> str:
    name = _unique("trainer")
    env = {
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "N_ESTIMATORS": str(req.n_estimators),
        "MAX_DEPTH": str(req.max_depth),
        "MLFLOW_EXPERIMENT": req.experiment,
        "REGISTER_MODEL_NAME": req.register_model_name,
    }
    if req.set_alias:
        env["SET_ALIAS"] = req.set_alias
    docker_client.containers.run(
        image=TRAINER_IMAGE, name=name, detach=True, network=network, environment=env, labels={"app":"trainer"}
    )
    return name

def _wait_trainer(name: str, wait_seconds: int) -> tuple[int, str]:
    status, logs_text = 1, ""
    try:
        c = docker_client.containers.get(name)
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

def _parse_last_json_line(text: str) -> dict | None:
    for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except Exception:
            continue
    return None

# ---------- FastAPI ----------
app = FastAPI(title="Router (blue/green via Caddy proxy)", version="4.0.0")

@app.get("/status", response_model=StatusResp)
def status() -> StatusResp:
    s = _load_state()
    internal = s.get("url")
    # Public URL is always the fixed proxy address
    public_url = f"http://{PUBLIC_HOST}:{PROXY_PUBLIC_PORT}"
    return StatusResp(
        active=s.get("active"),
        url=internal,
        public_url=public_url,
        healthy=_ping(internal) if internal else False,
        ts=s.get("ts"),
    )

@app.post("/admin/roll", response_model=RollResp)
def roll(body: RollReq):
    """
    Start candidate (internal only) -> health -> flip proxy -> drain -> retire old.
    """
    target_uri = _resolve_models_uri(body.name, body.ref)

    # 1) start candidate
    candidate_name = _unique("serve-cand")
    _start_generic_runtime(candidate_name, COMPOSE_NETWORK, target_uri)
    cand_internal = f"http://{candidate_name}:{SERVE_PORT}"

    # 2) health-check candidate
    deadline = time.time() + body.wait_ready_seconds
    while time.time() < deadline:
        if _ping(cand_internal):
            break
        time.sleep(0.5)
    else:
        _retire(candidate_name)
        raise HTTPException(status_code=503, detail="candidate not healthy")

    # 3) atomically flip proxy to candidate
    _proxy_point_to(candidate_name)

    # 4) optional drain grace (let old in-flight finish)
    time.sleep(DRAIN_GRACE_SEC)

    # 5) retire old
    state = _load_state()
    old_name = state.get("active")
    _retire(old_name)

    # 6) update state (public URL is fixed proxy URL)
    s = {
        "active": candidate_name,
        "url": cand_internal,
        "public_url": f"http://{PUBLIC_HOST}:{PROXY_PUBLIC_PORT}",
        "model_uri": target_uri,
        "ts": time.time(),
    }
    _save_state(s)

    return RollResp(active=candidate_name, url=cand_internal, public_url=s["public_url"])

@app.post("/admin/train", response_model=TrainResp)
def admin_train(req: TrainReq):
    """
    Start trainer; wait; parse the last JSON line:
    {"run_id": "...", "registered_model":"Name","version":17,"alias":"production","metrics":{...}}
    """
    name = _start_trainer(req, COMPOSE_NETWORK)
    status, logs = _wait_trainer(name, req.wait_seconds)
    result = _parse_last_json_line(logs) or {}
    resp = TrainResp(
        container=name,
        run_id=result.get("run_id"),
        registered_model=result.get("registered_model"),
        version=result.get("version"),
        alias_set=result.get("alias"),
        metrics=result.get("metrics"),
    )
    print(f"Trainer exited with status {status}, version={resp.version}")
    if status != 0 and resp.version is None:
        resp.logs_tail = logs[-LOG_TAIL_ON_ERROR:]
        raise HTTPException(
            status_code=500,
            detail={"error": f"trainer failed (exit={status})", **resp.model_dump()},
        )
    return resp

@app.post("/admin/train_then_roll", response_model=TrainRollResp)
def admin_train_then_roll(req: TrainReq):
    """
    Train a new version, then roll to that exact version via the proxy.
    """
    train_resp = admin_train(req)
    version = train_resp.version
    name = req.register_model_name
    if not version or not name:
        raise HTTPException(status_code=500, detail="trainer did not return a version/registered_model")
    rolled = False
    try:
        _ = roll(RollReq(name=name, ref=version, wait_ready_seconds=HEALTH_TIMEOUT_SEC))
        rolled = True
    except HTTPException as e:
        return TrainRollResp(**train_resp.model_dump(), rolled=False, logs_tail=f"roll failed: {e.detail}")
    return TrainRollResp(**train_resp.model_dump(), rolled=rolled)
