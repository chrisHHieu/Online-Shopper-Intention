# The image ships code, not artifacts. Mount data/ and models/ from a run of
# fetch, features and train, or run those commands inside the container first.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    SHOPPER_INTENT_CONFIG=/app/configs/base.yaml

WORKDIR /app

# libgomp is required by the LightGBM native library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 appuser

COPY configs ./configs
COPY data/raw/.gitkeep ./data/raw/.gitkeep
COPY data/processed/.gitkeep ./data/processed/.gitkeep
COPY models/.gitkeep ./models/.gitkeep
COPY reports/.gitkeep ./reports/.gitkeep
COPY reports/figures/.gitkeep ./reports/figures/.gitkeep

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8125

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8125/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "shopper_intent.serving.app:app", "--host", "0.0.0.0", "--port", "8125"]
