FROM python:3.12-slim

# uv for fast installs
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install deps using lockfile (best cache hit)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Use the virtualenv that uv created
ENV PATH="/app/.venv/bin:${PATH}"

# 2) Copy application code (new structure)
COPY common.py config.py ./
COPY status ./status
COPY roll ./roll
COPY trainer ./trainer
COPY main.py .
COPY specs ./specs


EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
