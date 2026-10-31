"""Versioned model registry on disk.

Layout: ``models/<model_type>/<version>/{model.joblib,metadata.json}`` with a
plain text ``latest`` file per model type holding the version string.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from shopper_intent.models.model import IntentModel


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_version() -> str:
    """Timestamp based version, unique down to the second."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_revision(root: Path) -> tuple[str, bool]:
    """Short commit hash and dirty flag, or ``("unknown", False)`` outside git."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    return revision or "unknown", bool(status.strip())


def model_dir(models_dir: Path, model_type: str) -> Path:
    return models_dir / model_type


def version_dir(models_dir: Path, model_type: str, version: str) -> Path:
    return model_dir(models_dir, model_type) / version


def save(
    models_dir: Path,
    model: IntentModel,
    metadata: dict[str, Any],
    version: str | None = None,
) -> Path:
    """Write the artifact and metadata, then point ``latest`` at this version."""
    version = version or new_version()
    target = version_dir(models_dir, model.model_type, version)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / "model.joblib")
    payload = dict(metadata)
    payload.update({"version": version, "model_type": model.model_type})
    (target / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (model_dir(models_dir, model.model_type) / "latest").write_text(version + "\n", encoding="utf-8")
    return target


def latest_version(models_dir: Path, model_type: str) -> str:
    pointer = model_dir(models_dir, model_type) / "latest"
    if not pointer.is_file():
        raise FileNotFoundError(f"no 'latest' pointer for model type {model_type!r} in {models_dir}")
    version = pointer.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"empty 'latest' pointer for model type {model_type!r}")
    return version


def available_versions(models_dir: Path, model_type: str) -> list[str]:
    directory = model_dir(models_dir, model_type)
    if not directory.is_dir():
        return []
    return sorted(
        child.name
        for child in directory.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    )


def load(
    models_dir: Path,
    model_type: str,
    version: str | None = None,
) -> tuple[IntentModel, dict[str, Any]]:
    version = version or latest_version(models_dir, model_type)
    target = version_dir(models_dir, model_type, version)
    model_path = target / "model.joblib"
    metadata_path = target / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"artifact {model_type}/{version} is incomplete in {target}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def load_latest(models_dir: Path, model_type: str) -> tuple[IntentModel, dict[str, Any]]:
    return load(models_dir, model_type)
