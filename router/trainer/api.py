from __future__ import annotations
from typing import Optional, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from trainer.service import TrainerService

router = APIRouter(prefix="/admin")
_svc = TrainerService()


# Requests use spec; only optional wait override is exposed for now
class TrainReq(BaseModel):
    wait_seconds: Optional[int] = Field(default=None, ge=1, le=24 * 3600)


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
    public_url: str | None = None


@router.post("/train/{trainer}", response_model=TrainResp)
def admin_train(trainer: str, req: TrainReq):
    out: Dict[str, Any] = _svc.train(trainer, wait_seconds=req.wait_seconds)
    return TrainResp(**out)


@router.post("/train_then_roll/{trainer}", response_model=TrainRollResp)
def admin_train_then_roll(trainer: str, req: TrainReq):
    out: Dict[str, Any] = _svc.train_then_roll(trainer, wait_seconds=req.wait_seconds)
    return TrainRollResp(**out)
