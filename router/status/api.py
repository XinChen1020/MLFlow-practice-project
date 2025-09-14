from __future__ import annotations
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

import config as cfg
from common import load_state, ping

router = APIRouter()

class StatusResp(BaseModel):
    active: Optional[str] = None
    url: Optional[str] = None
    public_url: Optional[str] = None
    healthy: bool = False
    ts: Optional[float] = None

@router.get("/status", response_model=StatusResp)
def status() -> StatusResp:
    s = load_state()
    internal = s.get("url")
    public_url = f"http://{cfg.PUBLIC_HOST}:{cfg.PROXY_PUBLIC_PORT}"
    return StatusResp(
        active=s.get("active"),
        url=internal,
        public_url=public_url,
        healthy=ping(internal) if internal else False,
        ts=s.get("ts"),
    )