from __future__ import annotations
from typing import Union
from fastapi import APIRouter
from pydantic import BaseModel, Field

import config as cfg
from roll.service import RollService

router = APIRouter(prefix="/admin")
_svc = RollService()

class RollReq(BaseModel):
    name: str = Field(..., examples=["DiabetesRF"])
    ref: Union[str, int] = Field(..., examples=["@production", 17])
    wait_ready_seconds: int = Field(default=cfg.HEALTH_TIMEOUT_SEC, ge=1, le=900)

class RollResp(BaseModel):
    active: str
    url: str
    public_url: str

@router.post("/roll", response_model=RollResp)
def roll(body: RollReq):
    out = _svc.roll(name=body.name, ref=body.ref, wait_ready_seconds=body.wait_ready_seconds)
    return RollResp(**out)