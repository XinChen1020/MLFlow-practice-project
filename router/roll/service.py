from __future__ import annotations
import time, httpx
from typing import Any, Dict, Tuple, Union
from fastapi import HTTPException

import config as cfg
from common import ml_client, docker_client, load_state, save_state, unique, ping
from docker.types import Healthcheck



class RollService:
    def __init__(self, *, proxy_admin_url: str | None = None):
        self._ml = ml_client
        self._docker = docker_client
        self._admin_url = (proxy_admin_url or cfg.PROXY_ADMIN_URL).rstrip("/") + "/load"

    # --- public API ---
    def roll(
        self,
        *,
        name: str,
        ref: Union[str, int],
        wait_ready_seconds: int,
        serve_image: str | None = None,
    ) -> Dict[str, str]:
        target_uri, version = self._resolve_models_uri(name, ref)

        candidate_name = unique("serve-cand")
        self._start_runtime(candidate_name, 
                            cfg.COMPOSE_NETWORK, 
                            target_uri, 
                            serve_image=serve_image)
        cand_internal = f"http://{candidate_name}:{cfg.SERVE_PORT}"

        deadline = time.time() + wait_ready_seconds
        while time.time() < deadline:
            if ping(cand_internal):
                break
            time.sleep(0.5)
        else:
            self._retire(state_name=candidate_name)
            raise HTTPException(status_code=503, detail="candidate not healthy")

        self._proxy_point_to(candidate_name)
        time.sleep(cfg.DRAIN_GRACE_SEC)

        state = load_state()
        self._retire(state.get("active"))

        new_state = {
            "active": candidate_name,
            "url": cand_internal,
            "public_url": f"http://{cfg.PUBLIC_HOST}:{cfg.PROXY_PUBLIC_PORT}",
            "model_uri": target_uri,
            "model_version": version,
            "model_alias": cfg.PRODUCTION_ALIAS,
            "ts": time.time(),
        }
        save_state(new_state)

        try:
            self._ml.set_registered_model_alias(name, cfg.PRODUCTION_ALIAS, str(version))
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"failed to set alias '{cfg.PRODUCTION_ALIAS}' on model '{name}': {e}",
            )

        return {
            "active": candidate_name,
            "url": cand_internal,
            "public_url": new_state["public_url"],
            "model_uri": target_uri,
            "alias": cfg.PRODUCTION_ALIAS,
            "version": version,
        }

    # --- helpers ---
    def _resolve_models_uri(self, name: str, ref: Union[str, int]) -> Tuple[str, int]:
        try:
            if isinstance(ref, str) and ref.startswith("@"):  # alias
                mv = self._ml.get_model_version_by_alias(name, ref[1:])
                version = int(mv.version)
                return f"models:/{name}/{version}", version
            if isinstance(ref, int):
                version = int(ref)
                return f"models:/{name}/{version}", version
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"model/alias not found: {e}")
        raise HTTPException(status_code=400, detail="ref must be '@alias' or integer version")

    def _start_runtime(
        self,
        name: str,
        network: str,
        model_uri: str,
        *,
        serve_image: str | None = None,
    ) -> str:
        
        image = serve_image or cfg.SERVE_IMAGE
        if not image:
            raise HTTPException(
                status_code=500,
                detail="Serving image not configured; supply serve_image via the request or specs."
            )
        
        # Start a new container with healthcheck
        healthcheck = Healthcheck(
            test=["CMD", "curl", "--f",  f"http://localhost:{cfg.SERVE_PORT}/health"],
            interval=5 * 10**9,     # 5s in nanoseconds
            timeout=5 * 10**9,      # 5s
            start_period=5 * 10**9, # 5s
            retries=3,
        )

        container = self._docker.containers.run(
            image=image,
            name=name,
            detach=True,
            network=network,
            environment={
                "MLFLOW_TRACKING_URI": cfg.MLFLOW_TRACKING_URI,
                "SERVE_MODEL_URI": model_uri,
                "SERVE_PORT": str(cfg.SERVE_PORT)
            },
            restart_policy={"Name": "on-failure", "MaximumRetryCount": 1},
            labels={"app": "mlflow-serve"},
            healthcheck=healthcheck
        )
        return container.id

    def _retire(self, state_name: str | None) -> None:
        """
        Stop and remove a container by name, ignoring errors.
        """
        
        if not state_name:
            return
        try:
            c = self._docker.containers.get(state_name)
            c.stop(timeout=5) 
            c.remove()
        except Exception:
            print(f"Warning: failed to retire container {state_name}")

    def _make_caddy_config(self, target_container: str) -> Dict[str, Any]:
        """
        Make Caddy config to point to target container.
        """
        return {
            "admin": {"listen": ":2019"},
            "apps": {
                "http": {
                    "servers": {
                        "srv0": {
                            "listen": [f":{cfg.PROXY_PUBLIC_PORT}"],
                            "routes": [
                                {
                                    "handle": [
                                        {
                                            "handler": "reverse_proxy",
                                            "upstreams": [{"dial": f"{target_container}:{cfg.SERVE_PORT}"}],
                                        }
                                    ]
                                }
                            ],
                        }
                    }
                }
            },
        }

    def _proxy_point_to(self, target_container: str) -> None:
        """
        Point Caddy proxy to target container using POST.
        """
        
        cfg_json = self._make_caddy_config(target_container)
        r = httpx.post(self._admin_url, json=cfg_json, timeout=5.0)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"proxy switch failed: {e.response.text}")