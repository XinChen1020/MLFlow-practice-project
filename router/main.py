from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from docker.errors import NotFound

from common import docker_client, load_state, save_state
from roll.api import router as roll_router
from status.api import router as status_router
from trainer.api import router as trainer_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler to validate persisted rollout state."""

    state = load_state()
    active = state.get("active")

    if active:
        try:
            docker_client.containers.get(active)
        except NotFound:
            logger.info(
                "Persisted active container %s is missing; clearing active routing state.",
                active,
            )
            cleared_state = dict(state)
            cleared_state.update(
                {
                    "active": None,
                    "url": None,
                    "public_url": None,
                }
            )
            save_state(cleared_state)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to validate persisted active container %s: %s",
                active,
                exc,
                exc_info=True,
            )

    try:
        yield
    finally:
        # No shutdown-time cleanup; roll workflow handles retirement explicitly.
        return

app = FastAPI(title="Router (blue/green via Caddy proxy)", 
              version="0.1",
              lifespan=lifespan)
app.include_router(status_router)
app.include_router(roll_router)
app.include_router(trainer_router)

# This allows local development and not used in the Dockerfile
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)