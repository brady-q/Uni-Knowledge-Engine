# Python 3.12, not the host's 3.14: torch / sentence-transformers cp314 wheels
# are unreliable and would fall back to multi-hour source builds at M3.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# HF_HOME is where the embedding model's weights land. Set explicitly so
# compose can mount a volume over a known path - the default is buried under
# $HOME, and re-downloading 90 MB on every rebuild is a slow way to start.
# TOKENIZERS_PARALLELISM is off because the fast tokenizer forks workers and
# then warns on every call once the parent process has forked as well, which is
# exactly what `uvicorn --reload` does.
ENV HF_HOME=/home/appuser/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Dependencies first: this layer is cached until the pins actually change.
COPY requirements.txt requirements-dev.txt ./
ARG INSTALL_DEV=false
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install -r requirements-dev.txt; fi

COPY . .

# Non-root. /data and the model cache are created and chowned here so the named
# volumes mounted over them inherit the right ownership when Docker first
# populates them - a volume on a path that does not exist in the image is
# created owned by root, and the process cannot write to it.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/storage /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /data /home/appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
