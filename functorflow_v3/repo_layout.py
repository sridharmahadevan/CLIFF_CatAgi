"""Helpers for resolving optional local dependency paths in CLIFF_CatAgi."""

from __future__ import annotations

import os
import subprocess
import sys
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


def resolve_democritus_root() -> Path:
    env_value = os.environ.get("CLIFF_DEMOCRITUS_ROOT") or os.environ.get("DEMOCRITUS_REPO_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = (
        _REPO_ROOT / "third_party" / "Democritus_OpenAI",
        _WORKSPACE_ROOT / "Democritus_OpenAI",
    )
    return _first_existing(candidates, fallback=candidates[0])


def resolve_democritus_python(democritus_root: Path | None = None) -> Path:
    root = democritus_root.resolve() if democritus_root is not None else resolve_democritus_root()
    candidates = (
        *(Path(value).expanduser() for value in (
            os.environ.get("CLIFF_DEMOCRITUS_PYTHON"),
            os.environ.get("DEMOC_PYTHON"),
            os.environ.get("DEMOCRITUS_PYTHON"),
        ) if value),
        root / ".venv" / "bin" / "python3",
        root / ".venv" / "bin" / "python",
        root / ".venv_democritus" / "bin" / "python3",
        root / ".venv_democritus" / "bin" / "python",
        _REPO_ROOT / ".venv_cliff" / "bin" / "python3",
        _REPO_ROOT / ".venv_cliff" / "bin" / "python",
        _WORKSPACE_ROOT / ".venv_cliff" / "bin" / "python3",
        _WORKSPACE_ROOT / ".venv_cliff" / "bin" / "python",
        Path(sys.executable).resolve(),
    )
    for candidate in candidates:
        if candidate.exists() and _python_can_find_modules(candidate, ("tqdm",)):
            return candidate.absolute()
    return _first_existing(candidates, fallback=Path(sys.executable).resolve())


def _python_can_find_modules(python_path: Path, module_names: tuple[str, ...]) -> bool:
    code = (
        "import importlib.util, sys; "
        "mods = " + repr(tuple(module_names)) + "; "
        "sys.exit(0 if all(importlib.util.find_spec(name) is not None for name in mods) else 1)"
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return False
    return completed.returncode == 0


def resolve_cliff_python() -> Path:
    env_value = os.environ.get("CLIFF_PYTHON") or os.environ.get("CLIFF_VENV_PYTHON")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = (
        _REPO_ROOT / ".venv_cliff" / "bin" / "python3",
        _REPO_ROOT / ".venv_cliff" / "bin" / "python",
        _WORKSPACE_ROOT / ".venv_cliff" / "bin" / "python3",
        _WORKSPACE_ROOT / ".venv_cliff" / "bin" / "python",
    )
    return _first_existing(candidates, fallback=Path(sys.executable).resolve())


def resolve_cliff_site_packages() -> tuple[Path, ...]:
    env_value = os.environ.get("CLIFF_SITE_PACKAGES") or os.environ.get("CLIFF_VENV_SITE_PACKAGES")
    if env_value:
        return tuple(
            Path(part).expanduser().resolve()
            for part in env_value.split(os.pathsep)
            if part.strip()
        )
    candidates = (
        _REPO_ROOT / ".venv_cliff" / "lib" / "python3.11" / "site-packages",
        _REPO_ROOT / ".venv_cliff" / "lib" / "python3.10" / "site-packages",
        _REPO_ROOT / ".venv_cliff" / "lib" / "python3.9" / "site-packages",
        _WORKSPACE_ROOT / ".venv_cliff" / "lib" / "python3.11" / "site-packages",
        _WORKSPACE_ROOT / ".venv_cliff" / "lib" / "python3.10" / "site-packages",
        _WORKSPACE_ROOT / ".venv_cliff" / "lib" / "python3.9" / "site-packages",
    )
    return tuple(candidate.resolve() for candidate in candidates if candidate.exists())


def resolve_prometheus_root() -> Path:
    return _resolve_dependency_root(
        env_var="CLIFF_PROMETHEUS_ROOT",
        repo_local=_REPO_ROOT / "third_party" / "Prometheus_v1",
        sibling=_WORKSPACE_ROOT / "Prometheus_v1",
    )


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
