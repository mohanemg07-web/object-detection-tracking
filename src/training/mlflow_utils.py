"""MLflow / DagsHub setup helpers, isolated so training stays readable.

The tracking URI resolution order is:
  1. explicit ``tracking_uri`` argument,
  2. ``MLFLOW_TRACKING_URI`` env var,
  3. ``mlflow_tracking_uri`` from the train config,
  4. fall back to a local ``./mlruns`` directory.

DagsHub auth uses ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``
(the password is your DagsHub access token). We never hard-code secrets.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_tracking_uri(config_uri: str | None) -> str:
    """Resolve the MLflow tracking URI, defaulting to local ./mlruns."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    if config_uri and config_uri.strip():
        return config_uri.strip()
    local = Path("mlruns").resolve()
    return local.as_uri()


def setup_mlflow(tracking_uri: str, experiment: str) -> object | None:
    """Configure MLflow and return the module (or None if unavailable).

    Logging failures must never crash training, so this is defensive: if
    mlflow isn't installed or the server is unreachable we warn and return
    None, letting the caller skip logging.
    """
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed; skipping experiment tracking")
        return None

    is_dagshub = "dagshub.com" in tracking_uri
    if is_dagshub and not (
        os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD")
    ):
        print(
            "[mlflow] DagsHub URI detected but MLFLOW_TRACKING_USERNAME / "
            "MLFLOW_TRACKING_PASSWORD are not set. Set them to your DagsHub "
            "username and access token before training, or logging will fail."
        )

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        print(f"[mlflow] tracking -> {tracking_uri} (experiment: {experiment})")
        return mlflow
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] setup failed ({exc}); continuing without tracking")
        return None
