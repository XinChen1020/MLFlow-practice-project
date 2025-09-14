from fastapi import FastAPI
from status.api import router as status_router
from roll.api import router as roll_router
from trainer.api import router as trainer_router


app = FastAPI(title="Router (blue/green via Caddy proxy)", version="4.2.0")
app.include_router(status_router)
app.include_router(roll_router)
app.include_router(trainer_router)