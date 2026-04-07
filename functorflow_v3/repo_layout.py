"""Helpers for resolving optional local dependency paths in CLIFF_CatAgi."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent


def repo_root() -> Path:
    return _REPO_ROOT


def workspace_root() -> Path:
    return _WORKSPACE_ROOT


def resolve_basket_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_BASKET_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "BASKET",
        sibling=_WORKSPACE_ROOT / "BASKET",
    )


def resolve_brand_panel_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_BRAND_PANEL_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "brand_democritus_block_denoise",
        sibling=_WORKSPACE_ROOT / "brand_democritus_block_denoise",
    )


def resolve_brand_awareness_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_BRAND_AWARENESS_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "brand_awareness_democritus",
        sibling=_WORKSPACE_ROOT / "brand_awareness_democritus",
    )


def resolve_democritus_root() -> Path:
    env_value = os.environ.get("CLIFF_DEMOCRITUS_ROOT") or os.environ.get("DEMOCRITUS_REPO_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = (
        _REPO_ROOT / "third_party" / "Democritus_OpenAI",
        _WORKSPACE_ROOT / "Democritus_OpenAI",
    )
    return _first_existing(candidates, fallback=candidates[0])


def resolve_course_repo_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_COURSE_REPO_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "Category-Theory-for-AGI-UMass-CMPSCI-692CT",
        sibling=_WORKSPACE_ROOT / "Category-Theory-for-AGI-UMass-CMPSCI-692CT",
    )


def resolve_functorflow_julia_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_JULIA_REPO_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "FunctorFlow.jl",
        sibling=_WORKSPACE_ROOT / "FunctorFlow.jl",
    )


def resolve_julia_examples_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_JULIA_EXAMPLES_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "Julia FF",
        sibling=_WORKSPACE_ROOT / "Julia FF",
    )


def resolve_book_pdf_path() -> Path:
    env_value = os.environ.get("CLIFF_BOOK_PDF_PATH", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = (
        _REPO_ROOT / "catagi.pdf",
        _WORKSPACE_ROOT / "catagi.pdf",
    )
    return _first_existing(candidates, fallback=candidates[0])


def resolve_democritus_seed_pdf_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_DEMOCRITUS_PDF_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "FunctorFlow" / "data" / "democritus",
        sibling=_WORKSPACE_ROOT / "FunctorFlow" / "data" / "democritus",
    )


def _resolve_dependency_root(*, env_var: str, repo_local: Path, sibling: Path) -> Path:
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    if repo_local.exists():
        return repo_local.resolve()
    return sibling.resolve()


def _first_existing(candidates: tuple[Path, ...], *, fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return fallback.resolve()
