"""Download the raw UCI dataset and record what was received.

The raw payload is stored byte for byte so the hash in ``manifest.json`` can be
checked later. A cached file whose hash still matches its manifest is never
downloaded again unless ``force`` is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from shopper_intent.config import Settings

REQUIRED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
]

DATASET_TITLE = "UCI Online Shoppers Purchasing Intention"
DATASET_LICENSE = "CC BY 4.0 (UCI Machine Learning Repository, dataset 468)"


class FetchError(RuntimeError):
    """Raised when the raw dataset cannot be downloaded or looks wrong."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _native(value: Any) -> Any:
    if isinstance(value, float) and value != value:  # NaN
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _download(
    url: str,
    user_agent: str,
    timeout: int,
    max_retries: int,
    min_interval: float,
) -> bytes:
    """Fetch a URL with a fixed User-Agent, rate limit and exponential backoff."""
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/csv,application/csv,text/plain;q=0.9,*/*;q=0.5",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(min_interval * (2 ** (attempt - 1)))
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue
        if len(payload) < 10_000:
            last_error = FetchError(f"response too small from {url}: {len(payload)} bytes")
            continue
        if payload.lstrip()[:1] == b"<":
            last_error = FetchError(f"response from {url} looks like HTML, not CSV")
            continue
        return payload
    raise FetchError(f"could not download {url} after {max_retries + 1} attempts: {last_error}")


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _check_payload(payload: bytes, url: str, min_rows: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 - surface the parser error with context
        raise FetchError(f"payload from {url} is not parseable CSV: {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FetchError(f"payload from {url} is missing columns: {missing}")
    if len(frame) < min_rows:
        raise FetchError(f"payload from {url} has {len(frame)} rows, expected at least {min_rows}")
    return frame


def build_manifest(
    frame: pd.DataFrame,
    path: Path,
    source_url: str,
    primary_source_url: str,
    fallback_used: bool,
) -> dict[str, Any]:
    first = {key: _native(value) for key, value in frame.iloc[0].to_dict().items()}
    last = {key: _native(value) for key, value in frame.iloc[-1].to_dict().items()}
    return {
        "dataset": DATASET_TITLE,
        "source_url": source_url,
        "primary_source_url": primary_source_url,
        "fallback_used": fallback_used,
        "fetched_at": _now_iso(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "columns": [str(column) for column in frame.columns],
        "first_observation": first,
        "last_observation": last,
        "license": DATASET_LICENSE,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def cached_manifest(config: Settings) -> dict[str, Any] | None:
    """Return the manifest when the cached file still matches it, else ``None``."""
    if not config.raw_path.is_file() or not config.manifest_path.is_file():
        return None
    try:
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("sha256") != sha256_file(config.raw_path):
        return None
    expected = config.data.expected_sha256
    if expected and manifest.get("sha256") != expected:
        return None
    return manifest


def fetch(config: Settings, force: bool = False) -> dict[str, Any]:
    """Download the dataset if needed and return its manifest.

    Raises ``FetchError`` when neither the primary nor the fallback URL yields a
    payload that passes the column and row checks.
    """
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        manifest = cached_manifest(config)
        if manifest is not None:
            return manifest

    sources = [(config.data.source_url, False), (config.data.fallback_url, True)]
    errors: list[str] = []
    payload: bytes | None = None
    used_url = config.data.source_url
    used_fallback = False
    for url, is_fallback in sources:
        try:
            candidate = _download(
                url,
                user_agent=config.data.user_agent,
                timeout=config.data.timeout_seconds,
                max_retries=config.data.max_retries,
                min_interval=config.data.min_interval_seconds,
            )
            _check_payload(candidate, url, config.data.min_rows)
        except FetchError as exc:
            errors.append(str(exc))
            continue
        payload = candidate
        used_url = url
        used_fallback = is_fallback
        break
    if payload is None:
        raise FetchError("all sources failed: " + " | ".join(errors))

    tmp_path = config.raw_path.with_suffix(".csv.tmp")
    tmp_path.write_bytes(payload)
    frame = _read_frame(tmp_path)
    digest = sha256_file(tmp_path)
    expected = config.data.expected_sha256
    if expected and digest != expected:
        tmp_path.unlink(missing_ok=True)
        raise FetchError(f"sha256 mismatch: config pins {expected}, download was {digest}")
    os.replace(tmp_path, config.raw_path)

    manifest = build_manifest(
        frame=frame,
        path=config.raw_path,
        source_url=used_url,
        primary_source_url=config.data.source_url,
        fallback_used=used_fallback,
    )
    _write_manifest(config.manifest_path, manifest)
    return manifest


def load_raw(config: Settings) -> pd.DataFrame:
    """Read the cached raw CSV, failing with a clear message when it is absent."""
    if not config.raw_path.is_file():
        raise FileNotFoundError(f"raw data not found at {config.raw_path}; run 'fetch' first")
    return pd.read_csv(config.raw_path)
