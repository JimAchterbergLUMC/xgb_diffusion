"""Diffusion estimators backed by a fork of XGBoost."""

from importlib.metadata import PackageNotFoundError, version

from ._c_api import _py_version
from .xgbddpm import XGBDDPMClassifier, XGBDDPMRegressor
from .xgbdiffusion import XGBDiffusionRegressor

try:
    __version__ = version("xgb-diffusion")
except PackageNotFoundError:
    __version__ = _py_version()

__all__ = [
    "XGBDDPMClassifier",
    "XGBDDPMRegressor",
    "XGBDiffusionRegressor",
]
