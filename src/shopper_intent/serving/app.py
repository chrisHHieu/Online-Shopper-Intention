"""FastAPI service around the latest registered model.

The service loads ``models/<serving.model>/latest`` at startup and exposes the
stored threshold and metrics, so a caller can see which policy is being applied.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from shopper_intent import __version__
from shopper_intent.config import Settings, load_config
from shopper_intent.features.build import build_features
from shopper_intent.models import registry


class SessionIn(BaseModel):
    """One raw session row. Field names mirror the source CSV."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    administrative: int = Field(alias="Administrative", ge=0)
    administrative_duration: float = Field(alias="Administrative_Duration", ge=0)
    informational: int = Field(alias="Informational", ge=0)
    informational_duration: float = Field(alias="Informational_Duration", ge=0)
    product_related: int = Field(alias="ProductRelated", ge=0)
    product_related_duration: float = Field(alias="ProductRelated_Duration", ge=0)
    bounce_rates: float = Field(alias="BounceRates", ge=0, le=1)
    exit_rates: float = Field(alias="ExitRates", ge=0, le=1)
    page_values: float = Field(alias="PageValues", ge=0)
    special_day: float = Field(alias="SpecialDay", ge=0, le=1)
    month: str = Field(alias="Month")
    operating_systems: int = Field(alias="OperatingSystems", ge=1)
    browser: int = Field(alias="Browser", ge=1)
    region: int = Field(alias="Region", ge=1)
    traffic_type: int = Field(alias="TrafficType", ge=1)
    visitor_type: str = Field(alias="VisitorType")
    weekend: bool = Field(alias="Weekend")

    def to_frame(self, config: Settings) -> pd.DataFrame:
        record = self.model_dump(by_alias=True)
        for column, allowed in config.features.categorical.items():
            value = record[column]
            if value not in allowed:
                raise ValueError(f"{column}={value!r} is not in {allowed}")
        return pd.DataFrame([record])


class Contribution(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases", "decreases"]


class PredictionOut(BaseModel):
    conversion_probability: float
    decision: Literal["contact", "skip"]
    threshold: float
    expected_value_if_contacted: float
    expected_value_of_decision: float
    model_version: str
    top_contributions: list[Contribution]


class BatchIn(BaseModel):
    sessions: list[SessionIn] = Field(min_length=1, max_length=1000)


class BatchSummary(BaseModel):
    sessions: int
    contacts: int
    expected_conversions: float
    expected_net_value: float


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    summary: BatchSummary


def _prepare(application: FastAPI, frame: pd.DataFrame) -> pd.DataFrame:
    matrix = build_features(frame, application.state.config)
    names = application.state.feature_names
    if list(matrix.columns) != names:
        matrix = matrix.reindex(columns=names, fill_value=0.0)
    return matrix


def _predict(model: Any, frame: pd.DataFrame, top_k: int = 3) -> list[PredictionOut]:
    result = model.predict(frame)
    contributions = model.explain(frame, top_k=top_k)
    predictions: list[PredictionOut] = []
    for index in range(len(frame)):
        predictions.append(
            PredictionOut(
                conversion_probability=round(float(result["probability"][index]), 6),
                decision=str(result["decision"][index]),
                threshold=round(float(model.threshold), 6),
                expected_value_if_contacted=round(
                    float(result["expected_value_if_contacted"][index]), 4
                ),
                expected_value_of_decision=round(
                    float(result["expected_value_of_decision"][index]), 4
                ),
                model_version=model.version,
                top_contributions=[Contribution(**row) for row in contributions[index]],
            )
        )
    return predictions


def create_app(config: Settings | None = None) -> FastAPI:
    settings = config or load_config()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.config = settings
        application.state.model = None
        application.state.metadata = None
        application.state.feature_names = []
        application.state.load_error = None
        try:
            model, metadata = registry.load_latest(settings.models_dir, settings.serving.model)
            application.state.model = model
            application.state.metadata = metadata
            application.state.feature_names = list(metadata.get("features") or model.feature_names)
        except Exception as exc:  # noqa: BLE001 - the service stays up and reports the reason
            application.state.load_error = str(exc)
        yield

    application = FastAPI(
        title="Online shopper intention",
        description="Calibrated session-level purchase intention with a value-based threshold.",
        version=__version__,
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> JSONResponse:
        model = getattr(application.state, "model", None)
        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "model_loaded": False,
                    "error": application.state.load_error,
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "model_loaded": True,
                "model_type": model.model_type,
                "model_version": model.version,
            },
        )

    @application.get("/model/info")
    def model_info() -> dict[str, Any]:
        model = getattr(application.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail=application.state.load_error)
        metadata = application.state.metadata or {}
        return {
            **model.info(),
            "trained_at": model.trained_at,
            "git_commit": metadata.get("git_commit"),
            "metrics": metadata.get("metrics", {}),
            "protocol": metadata.get("protocol"),
            "feature_names": model.feature_names,
        }

    @application.post("/predict", response_model=PredictionOut)
    def predict(session: SessionIn) -> PredictionOut:
        model = getattr(application.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail=application.state.load_error)
        try:
            frame = _prepare(application, session.to_frame(settings))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _predict(model, frame)[0]

    @application.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn) -> BatchOut:
        model = getattr(application.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail=application.state.load_error)
        try:
            raw = pd.concat(
                [session.to_frame(settings) for session in batch.sessions], ignore_index=True
            )
            frame = _prepare(application, raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        predictions = _predict(model, frame)
        contacts = sum(1 for prediction in predictions if prediction.decision == "contact")
        summary = BatchSummary(
            sessions=len(predictions),
            contacts=contacts,
            expected_conversions=round(
                sum(prediction.conversion_probability for prediction in predictions), 4
            ),
            expected_net_value=round(
                sum(prediction.expected_value_of_decision for prediction in predictions), 4
            ),
        )
        return BatchOut(predictions=predictions, summary=summary)

    return application


app = create_app()
