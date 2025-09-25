import os, json, glob, yaml

# ---------- Core env ----------
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DOCKER_HOST          = os.getenv("DOCKER_HOST", "http://socket-proxy:2375")

# Generic serving runtime (fallbacks used by roll if tags/specs are missing)
SERVE_IMAGE          = os.getenv("SERVE_IMAGE")
SERVE_PORT           = int(os.getenv("SERVE_PORT", "8080"))

# State & network
STATE_PATH           = os.getenv("STATE_PATH", "/state/active.json")
COMPOSE_NETWORK      = os.getenv("COMPOSE_NETWORK", f"{os.getenv('COMPOSE_PROJECT_NAME', 'local')}_default")
HEALTH_TIMEOUT_SEC   = int(os.getenv("HEALTH_TIMEOUT_SEC", "90"))
DRAIN_GRACE_SEC      = float(os.getenv("DRAIN_GRACE_SEC", "1.5"))

# Trainer defaults (used if spec omits fields)
TRAINER_IMAGE        = os.getenv("TRAINER_IMAGE")
TRAINER_TIMEOUT_SEC  = int(os.getenv("TRAINER_TIMEOUT_SEC", "1800"))
LOG_TAIL_ON_ERROR    = int(os.getenv("TRAINER_LOG_TAIL", "1200"))

# Caddy reverse proxy
PROXY_PUBLIC_PORT    = int(os.getenv("PROXY_PUBLIC_PORT", "9000"))
PROXY_ADMIN_URL      = os.getenv("PROXY_ADMIN_URL", "http://proxy:2019")
PUBLIC_HOST          = os.getenv("PUBLIC_HOST", "localhost")

# Trainer specs (directory or single file)
SPECS_PATH   = os.getenv("SPECS_PATH")  # e.g., /app/trainer-specs

def _read_text(path: str) -> str:
    """Read file and expand ${VARS} from environment (Compose-friendly)."""
    with open(path, "r") as f:
        return os.path.expandvars(f.read())

def load_specs() -> dict:
    """
    Load trainer specs from a directory (merging *.yaml|*.yml|*.json) or a single file.
    Each file may contain one or more trainer keys at top level.
    """
    if not SPECS_PATH:
        return {}

    # Directory: merge all files
    if os.path.isdir(SPECS_PATH):
        out: dict = {}
        for fp in sorted(glob.glob(os.path.join(SPECS_PATH, "*.*"))):
            try:
                text = _read_text(fp)
                if fp.endswith((".yaml", ".yml")):
                    spec = yaml.safe_load(text) or {}
                else:
                    spec = json.loads(text)
                if not isinstance(spec, dict):
                    continue
                out.update(spec)  # shallow merge by trainer key
            except Exception:
                # Ignore bad files; a validator in service will catch missing keys later
                continue
        return out

    # Single file: parse by extension
    try:
        text = _read_text(SPECS_PATH)
        if SPECS_PATH.endswith((".yaml", ".yml")):
            return yaml.safe_load(text) or {}
        return json.loads(text)
    except Exception:
        return {}
