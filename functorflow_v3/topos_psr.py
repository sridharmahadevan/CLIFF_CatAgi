"""Reuse the shared topos-PSR implementation from the FunctorFlow_v3 package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_shared_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "FunctorFlow_v3"
        / "functorflow_v3"
        / "topos_psr.py"
    )
    spec = importlib.util.spec_from_file_location("shared_functorflow_topos_psr", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared topos_psr module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SHARED_MODULE = _load_shared_module()

build_review_episodes = _SHARED_MODULE.build_review_episodes
build_topos_psr_bundle = _SHARED_MODULE.build_topos_psr_bundle

__all__ = ("build_review_episodes", "build_topos_psr_bundle")
