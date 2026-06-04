"""Tests for training helpers that don't require torch/ultralytics."""

import os
from unittest import mock

from src.training.mlflow_utils import resolve_tracking_uri


def test_explicit_config_uri_used_when_no_env():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        uri = resolve_tracking_uri("https://dagshub.com/u/r.mlflow")
        assert uri == "https://dagshub.com/u/r.mlflow"


def test_env_var_overrides_config():
    with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://env-server"}):
        uri = resolve_tracking_uri("https://dagshub.com/u/r.mlflow")
        assert uri == "http://env-server"


def test_falls_back_to_local_mlruns():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        uri = resolve_tracking_uri("")
        assert uri.startswith("file:")
        assert uri.rstrip("/").endswith("mlruns")


def test_blank_config_uri_treated_as_empty():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        uri = resolve_tracking_uri("   ")
        assert uri.startswith("file:")
