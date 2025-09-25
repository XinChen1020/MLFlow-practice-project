# Router Admin API

The router service exposes a small admin surface for launching model trainers and
rolling out the resulting models. Trainer behavior is defined via YAML or JSON
specifications in `trainer-specs/`.

## Trainer specifications

Each top-level key in a trainer spec file corresponds to a trainer name passed to
`/admin/train/{trainer}`. The schema supports optional overrides for different
container images:

```yaml
<trainer-name>:
  trainer_image: example/trainer:latest   # default trainer image when no selector is given
  image_options:                          # optional selector -> trainer image mapping
    cpu: example/trainer:cpu
    gpu: example/trainer:cuda
  serve_image: example/server:latest      # optional default serving runtime image
  serve_image_options:                    # optional selector -> serving image mapping
    cpu: example/server:cpu
    gpu: example/server:cuda
  timeout: 1800
  gpus: "all"
  env:
    MLFLOW_EXPERIMENT: my_experiment
```

When a request specifies an `image_key`, the service swaps in the corresponding
image from `image_options`. Omitting the key keeps the `trainer_image` default.
Supplying an unknown key results in HTTP 400 with the list of supported keys.

## Admin trainer endpoints

Two POST endpoints accept the same request payload:

- `POST /admin/train/{trainer}` launches the trainer container and waits for
  completion.
- `POST /admin/train_then_roll/{trainer}` runs the trainer and, if successful,
  rolls out the produced model version.

Request body fields:

- `wait_seconds` (optional): override the spec or default timeout.
- `image_key` (optional): select one of the spec's `image_options` entries.
- `parameters` (optional): dictionary of environment variable overrides passed to
  the trainer container. Values are serialized as strings; setting a value to
  `null` removes it from the final environment.

Responses include the resolved `image_key` (or `null` when the default image was
used), the serving image that will be used for rollout, and any applied
parameter overrides so logs and API clients can confirm how the trainer ran.

The `POST /admin/roll` endpoint accepts an optional `serve_image` field when you
want to override the runtime on a per-request basis. When the field is omitted,
the router falls back to the serving image resolved from the trainer spec (or
from the `SERVE_IMAGE` environment variable for backwards compatibility).