# Python 3.12, not the host's 3.14: torch / sentence-transformers cp314 wheels
# are unreliable and would fall back to multi-hour source builds at M3.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first: this layer is cached until the pins actually change.
COPY requirements.txt requirements-dev.txt ./
ARG INSTALL_DEV=false
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install -r requirements-dev.txt; fi

COPY . .

# Non-root. /data is created and chowned here so the named storage volume
# inherits the right ownership when Docker first populates it.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/storage \
    && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
