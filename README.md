# MLFlow Switch Orchestrator

MLFlow Switch Orchestrator provides a control plane for training and blue/green rolling deployments of MLflow-registered models. A FastAPI router coordinates on-demand trainers, manages candidate containers, and flips a Caddy reverse proxy once health checks pass.

## System Overview

- **Router service (`router/`)** – FastAPI admin surface that validates persisted rollout state on startup, launches trainer containers, registers model versions in MLflow, and rolls new revisions by updating the proxy and MLflow aliases.
- **Trainer containers (`model-trainer/`)** – Spec-driven workloads that receive environment overrides, log lineage/metrics to MLflow, and optionally trigger immediate rollout.
- **Serve containers** – Runtime images (e.g., the reference scikit-learn inference image) that pull the promoted model artifacts from MLflow and expose inference APIs once traffic is flipped to them.
- **MLflow tracking (`docker-compose.prod.yaml`)** – Local MLflow server with SQLite backend store and file-based artifacts used by both trainers and the router.
- **Caddy proxy + socket proxy** – Caddy exposes a stable public port while the router talks to Docker through a restricted socket proxy for safer automation.

## Getting Started
1. Create a `.env` with any overrides for the variables referenced in `docker-compose.prod.yaml` (defaults work for local testing).
2. Build the reference trainer and serving images that power the bundled `sklearn-model-1` spec using the provided compose file (it tags both images as `:latest` by default):
   ```bash
   docker compose -f model-trainer/sklearn-model-1/docker-compose.prod.yml build
   ```
3. Launch the stack:
   ```bash
   docker compose -f docker-compose.prod.yaml up --build
   ```
4. Trigger training or a train-then-rollout via the admin API (example uses the bundled `sklearn-model-1` spec):
   ```bash
   curl -X POST \
     'http://localhost:8000/admin/train_then_roll/sklearn-model-1' \
     -H 'Content-Type: application/json' \
     -d '{"wait_seconds": 600, "parameters": {"N_ESTIMATORS": 256}}'
   ```
5. Once the rollout completes, call the active MLflow inference server (proxied on port `9000`) to score requests. The payload below mirrors `test_script.sh` and uses `dataframe_split` formatting:
   ```bash
   curl -X POST \
     'http://localhost:9000/invocations' \
     -H 'Content-Type: application/json' \
     -d '{
       "dataframe_split": {
         "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
         "data": [[0.03, 1, 0.06, 0.03, 0.04, 0.03, 0.02, 0.03, 0.04, 0.01]]
       }
     }'
   ```
6. Check router status and the active public endpoint:
   ```bash
   curl http://localhost:8000/status
   ```
7. Inspect experiment runs and registered models in MLflow at `http://localhost:${MLFLOW_SERVICE_PORT}` (defaults to `http://localhost:9010` unless overridden in your `.env`).

## Creating Your Own Trainer and Serving Images
1. **Duplicate the sample project** – Copy `model-trainer/sklearn-model-1/` to a new folder and adjust its source to prepare data, train, and log to MLflow the way your model requires.
2. **Author Dockerfiles + compose entry** – Update the `docker_build/` Dockerfiles (or create new ones) to install dependencies and define the entrypoints for training (`Dockerfile_training`) and serving (`Dockerfile_serve`). Extend `docker-compose.prod.yml` (or create a sibling compose file) so `docker compose build` produces the trainer and serving images with tags that match what you plan to reference in router specs.
3. **Register the spec** – Add a new entry to `router/specs/spec.yaml` pointing at your trainer and server images, default environment variables, timeouts, and any image selectors you need:
   ```yaml
   my-new-model:
     trainer_image: trainer-my-model:latest
     serve_image: server-my-model:latest
     timeout: 3600
     env:
       REGISTERED_MODEL_NAME: MyCoolModel
       MLFLOW_EXPERIMENT: my_experiment
   ```
4. **Trigger via API** – Call `/admin/train/{spec-name}` to launch training or `/admin/train_then_roll/{spec-name}` to train and deploy. The router injects MLflow credentials/IDs and, on success, can immediately roll out your serving image.

Serving-only rollouts can reuse existing registry entries by invoking `/admin/roll` with a model name and version or alias.

## API Quick Reference

| Endpoint | Method | Description | Example |
| --- | --- | --- | --- |
| `/status` | `GET` | Returns the active container ID, internal URL, public proxy URL, and health indicator. | `curl http://localhost:8000/status` |
| `/admin/train/{trainer}` | `POST` | Launches the trainer defined by `{trainer}`. Accepts optional `wait_seconds`, `image_key`, and `parameters` overrides. | `curl -X POST http://localhost:8000/admin/train/sklearn-model-1 -H 'Content-Type: application/json' -d '{"parameters":{"N_ESTIMATORS":128}}'` |
| `/admin/train_then_roll/{trainer}` | `POST` | Runs training and, on success, deploys the produced model using the configured serving image. | `curl -X POST http://localhost:8000/admin/train_then_roll/sklearn-model-1 -H 'Content-Type: application/json' -d '{"wait_seconds":600}'` |
| `/admin/roll` | `POST` | Promotes an existing MLflow model version or alias into production without retraining. | `curl -X POST http://localhost:8000/admin/roll -H 'Content-Type: application/json' -d '{"name":"DiabetesRF","ref":"@staging"}'` |

## Design Notes
- Trainer specs (`router/specs/`) map trainer names to Docker images, optional image selectors, timeouts, GPU needs, and environment variables. Requests may add overrides or choose alternate images.
- Before a trainer container starts, the router pre-creates an MLflow run and injects `MLFLOW_RUN_ID`, giving reliable lineage between automation and the registry entry.
- Completed trainers optionally trigger the rollout service, which stands up the candidate container, waits for health checks, flips the proxy to the new target, and records the active container ID on disk for crash-safe recovery.
- The rollout API can also promote an existing registered model version or alias without retraining, keeping model deployment and training concerns loosely coupled.

## Example Directory Layout
```
router/
├── main.py              # FastAPI app wiring status, trainer, and rollout routers
├── specs/spec.yaml      # Sample trainer specification(s)
├── trainer/             # Admin endpoints + Docker orchestration helpers
├── roll/                # Blue/green deployment service and health checks
└── status/              # Active deployment status endpoint
model-trainer/
└── sklearn-model-1/     # Reference trainer + server build contexts
```

## To Do
- [x] Document architecture, quickstart workflow, and design decisions for the orchestrator.
- [x] Publish example admin API usage for training and rollout flows.
- [x] Ship the reference scikit-learn trainer + inference images that back the `sklearn-model-1` spec.
- [ ] Add PyTorch + Hugging Face example trainer and serving images.
- [ ] Build a simple UI to visualize training progress and the active deployment slot.
- [ ] Integrate Apache Airflow for time- or performance-triggered retraining workflows.
- [ ] Improve streaming data ingestion/management strategy for near-real-time retraining.
- [ ] Add unit test coverage for critical router and trainer components.
- [ ] Introduce smoke tests to validate end-to-end deployment flows after changes.
- [ ] Allows datasource other than local (such as AWS S3)
