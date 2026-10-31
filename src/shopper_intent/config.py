"""Typed configuration loaded from YAML.

The config file is the single place where data paths, model hyperparameters and
the business value of a conversion are defined. Relative paths resolve against
the project root, which is the directory holding ``pyproject.toml`` unless the
config file itself lives in a ``configs/`` directory.

Every section forbids unknown keys: a typo in a YAML file is an error, not a
silently ignored option.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_ENV_VAR = "SHOPPER_INTENT_CONFIG"
ROOT_ENV_VAR = "SHOPPER_INTENT_ROOT"

ModelName = Literal["logistic", "lightgbm", "xgboost", "catboost"]
MODEL_NAMES: tuple[str, ...] = ("logistic", "lightgbm", "xgboost", "catboost")


class Section(BaseModel):
    """Base for config sections: unknown keys are an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ProjectSettings(Section):
    name: str
    version: str


class DataSettings(Section):
    source_url: str
    fallback_url: str
    expected_sha256: str | None = None
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    raw_filename: str = "online_shoppers_intention.csv"
    user_agent: str
    timeout_seconds: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    min_interval_seconds: float = Field(default=1.0, ge=0)
    min_rows: int = Field(default=1000, ge=1)
    max_duplicate_fraction: float = Field(default=0.02, ge=0, le=1)
    min_positive_rate: float = Field(default=0.01, gt=0, lt=1)
    max_positive_rate: float = Field(default=0.99, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_rates(self) -> DataSettings:
        if self.min_positive_rate >= self.max_positive_rate:
            raise ValueError("min_positive_rate must be below max_positive_rate")
        return self


class FeatureSettings(Section):
    target: str = "Revenue"
    processed_filename: str = "sessions.csv"
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    seed: int = 42
    numeric: list[str]
    categorical: dict[str, list[str | bool]]

    @model_validator(mode="after")
    def _check_columns(self) -> FeatureSettings:
        if not self.numeric:
            raise ValueError("numeric feature list is empty")
        overlap = set(self.numeric) & set(self.categorical)
        if overlap:
            raise ValueError(f"columns listed as both numeric and categorical: {sorted(overlap)}")
        if self.target in self.numeric or self.target in self.categorical:
            raise ValueError("target column must not be a feature")
        return self


class LightGBMSettings(Section):
    n_estimators: int = Field(default=400, ge=10)
    learning_rate: float = Field(default=0.05, gt=0, lt=1)
    num_leaves: int = Field(default=31, ge=2)
    min_child_samples: int = Field(default=40, ge=1)
    subsample: float = Field(default=0.9, gt=0, le=1)
    subsample_freq: int = Field(default=1, ge=0)
    colsample_bytree: float = Field(default=0.9, gt=0, le=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    is_unbalance: bool = True


class LogisticSettings(Section):
    C: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=2000, ge=100)
    class_weight: Literal["balanced", "none"] = "balanced"


class XGBoostSettings(Section):
    """Native categorical boosting.

    ``scale_pos_weight="auto"`` resolves to the negative to positive ratio of
    the training fold at fit time; a number pins it. ``tree_method="hist"`` is
    the only supported value because the exact method is slow on this table and
    the GPU methods are not available in CI.
    """

    n_estimators: int = Field(default=700, ge=10)
    max_depth: int = Field(default=5, ge=2, le=16)
    learning_rate: float = Field(default=0.05, gt=0, lt=1)
    subsample: float = Field(default=0.9, gt=0, le=1)
    colsample_bytree: float = Field(default=0.9, gt=0, le=1)
    min_child_weight: float = Field(default=1.0, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    tree_method: Literal["hist"] = "hist"
    enable_categorical: bool = True
    scale_pos_weight: float | Literal["auto"] = "auto"


class CatBoostSettings(Section):
    """Native categorical boosting.

    No monotone constraints: the business relationship between every feature and
    conversion has not been established, and imposing one without evidence would
    be a modelling claim the data does not support.

    ``boosting_type="Plain"`` and ``one_hot_max_size=16`` are a speed choice,
    not a modelling one: the ordered scheme and target statistics on the three
    low cardinality categoricals cost roughly seven times the fit time here and
    scored lower on a single holdout check.
    """

    iterations: int = Field(default=500, ge=10)
    depth: int = Field(default=6, ge=2, le=10)
    learning_rate: float = Field(default=0.05, gt=0, lt=1)
    l2_leaf_reg: float = Field(default=3.0, ge=0)
    auto_class_weights: Literal["Balanced", "None"] = "Balanced"
    eval_metric: Literal["PRAUC"] = "PRAUC"
    boosting_type: Literal["Plain", "Ordered"] = "Plain"
    one_hot_max_size: int = Field(default=16, ge=2)
    bootstrap_type: Literal["Bayesian", "Bernoulli"] = "Bayesian"
    subsample: float = Field(default=0.9, gt=0, le=1)
    random_strength: float = Field(default=1.0, ge=0)
    thread_count: int = Field(default=4, ge=1)
    verbose: bool = False


class ModelSettings(Section):
    seed: int = 42
    cv_folds: int = Field(default=5, ge=2, le=10)
    # The two cheap models are the default so CI stays light; configs/full.yaml
    # turns on all four.
    candidates: list[ModelName] = Field(default_factory=lambda: ["logistic", "lightgbm"])
    lightgbm: LightGBMSettings = LightGBMSettings()
    logistic: LogisticSettings = LogisticSettings()
    xgboost: XGBoostSettings = XGBoostSettings()
    catboost: CatBoostSettings = CatBoostSettings()

    @field_validator("candidates")
    @classmethod
    def _check_candidates(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("model.candidates must name at least one model")
        if len(set(values)) != len(values):
            raise ValueError("model.candidates must not repeat a model")
        return values


class BusinessSettings(Section):
    value_per_conversion: float = Field(default=100.0, gt=0)
    cost_per_outreach: float = Field(default=1.0, gt=0)
    default_threshold: float = Field(default=0.5, gt=0, lt=1)
    threshold_candidates: int = Field(default=500, ge=20, le=20000)
    sensitivity_costs: list[float] = Field(
        default_factory=lambda: [1.0, 5.0, 10.0, 25.0, 50.0],
        min_length=1,
    )

    @field_validator("sensitivity_costs")
    @classmethod
    def _check_sensitivity_costs(cls, values: list[float]) -> list[float]:
        if any(value <= 0 for value in values):
            raise ValueError("sensitivity costs must be positive")
        if len(set(values)) != len(values):
            raise ValueError("sensitivity costs must be unique")
        return values


class CalibrationSettings(Section):
    """Post-hoc probability calibration.

    ``auto`` fits both isotonic and Platt on the out-of-fold training scores and
    keeps whichever has the lower out-of-fold Brier score. ``isotonic`` and
    ``platt`` force one method, which is what the earlier versions of this
    repository did with isotonic only.
    """

    method: Literal["isotonic", "platt", "auto"] = "auto"
    clip_low: float = Field(default=0.0001, ge=0, lt=0.5)
    clip_high: float = Field(default=0.9999, gt=0.5, le=1)

    @model_validator(mode="after")
    def _check_clip(self) -> CalibrationSettings:
        if self.clip_low >= self.clip_high:
            raise ValueError("clip_low must be below clip_high")
        return self


class ConformalSettings(Section):
    """Split conformal prediction sets.

    ``alpha`` is the miscoverage rate: ``0.1`` asks for at least 90 percent
    coverage on exchangeable data. Sets are built from out-of-fold training
    scores and measured once on the held-out split.
    """

    enabled: bool = True
    alpha: float = Field(default=0.1, gt=0, lt=1)


class TuningSettings(Section):
    trials: int = Field(default=40, ge=1, le=1000)
    fast_trials: int = Field(default=5, ge=1, le=100)
    timeout_seconds: int = Field(default=1800, ge=10)
    sampler_seed: int = 42
    storage: str = "reports/optuna.db"
    study_prefix: str = "shopper-intent"
    n_startup_trials: int = Field(default=10, ge=1)
    n_warmup_steps: int = Field(default=1, ge=0)
    # Tunable only on the model named by serving.model unless --models widens it.
    models: list[ModelName] | None = None


class TrackingSettings(Section):
    """MLflow tracking and registry. Disabled unless ``train --track`` is passed."""

    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment: str = "shopper-intent"
    registered_model_prefix: str = "shopper_intent"
    champion_alias: str = "champion"
    artifact_subdir: str = "model"


class RegistrySettings(Section):
    models_dir: str = "models"
    reports_dir: str = "reports"


class ServingSettings(Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8125, ge=1, le=65535)
    model: ModelName = "lightgbm"


class Settings(Section):
    """Full configuration plus the resolved project root."""

    project: ProjectSettings
    data: DataSettings
    features: FeatureSettings
    model: ModelSettings
    business: BusinessSettings
    calibration: CalibrationSettings
    conformal: ConformalSettings = ConformalSettings()
    tuning: TuningSettings = TuningSettings()
    tracking: TrackingSettings = TrackingSettings()
    registry: RegistrySettings
    serving: ServingSettings
    root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def _check_serving_model(self) -> Settings:
        if self.serving.model not in self.model.candidates:
            raise ValueError(
                f"serving.model {self.serving.model!r} is not in model.candidates "
                f"{self.model.candidates}; the service would have no artifact to load"
            )
        return self

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.data.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.data.processed_dir)

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.data.raw_filename

    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / "manifest.json"

    @property
    def processed_path(self) -> Path:
        return self.processed_dir / self.features.processed_filename

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.registry.models_dir)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.registry.reports_dir)

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def optuna_storage(self) -> Path:
        return self.resolve(self.tuning.storage)

    @property
    def mlruns_dir(self) -> Path:
        uri = self.tracking.tracking_uri
        prefix = "file:"
        tail = uri[len(prefix):] if uri.startswith(prefix) else uri
        return self.resolve(tail)

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path)

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"root", "source_path"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def project_root(start: Path | None = None) -> Path:
    """Find the repository root: env override, then pyproject.toml, then package location."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


def _resolve_root(config_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def load_config(
    path: str | Path | None = None,
    root: str | Path | None = None,
) -> Settings:
    """Load and validate a YAML config.

    Resolution order for ``path``: explicit argument, ``SHOPPER_INTENT_CONFIG``,
    then ``<project root>/configs/base.yaml``.
    """
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR) or (project_root() / "configs" / "base.yaml")
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")
    resolved_root = _resolve_root(config_path, Path(root) if root is not None else None)
    return Settings(**raw, root=resolved_root, source_path=config_path)
