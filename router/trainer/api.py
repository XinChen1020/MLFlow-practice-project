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
    image_key: Optional[str] = Field(
        default=None,
        description="Optional selector matching the trainer spec image_options mapping.",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional mapping of environment parameter overrides for the trainer run.",
    )


class TrainResp(BaseModel):
    container: str
    run_id: str | None = None
    registered_model: str | None = None
    version: int | None = None
    alias_set: str | None = None
    model_uri: str | None = None
    metrics: dict | None = None
    logs_tail: str | None = None
    image_key: str | None = Field(default=None, description="Image key used for the run.")
    serve_image: str | None = Field(
        default=None,
        description="Serving image resolved for rolling out the produced model.",
    )
    parameters: Dict[str, Any] | None = Field(
        default=None,
        description="Applied parameter overrides for the trainer run.",
    )


class TrainRollResp(TrainResp):
    rolled: bool = False
    public_url: str | None = None


@router.post("/train/{trainer}", response_model=TrainResp)
def admin_train(trainer: str, req: TrainReq):
    out: Dict[str, Any] = _svc.train(
        trainer,
        wait_seconds=req.wait_seconds,
        image_key=req.image_key,
        parameters=req.parameters,

    )
    return TrainResp(**out)


@router.post("/train_then_roll/{trainer}", response_model=TrainRollResp)
def admin_train_then_roll(trainer: str, req: TrainReq):
    out: Dict[str, Any] = _svc.train_then_roll(

        trainer,
        wait_seconds=req.wait_seconds,
        image_key=req.image_key,
        parameters=req.parameters,
    )
    return TrainRollResp(**out)
