FROM python:3.12-slim

# Pin a specific uv version tag once the project stabilises
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/.models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) dependencies (cached layer). Uses the lockfile when present.
COPY pyproject.toml .python-version uv.lock* ./
RUN if [ -f uv.lock ]; then uv sync --frozen --no-dev; else uv sync --no-dev; fi

# 2) Week 1+: bake embedding models into the image so containers never download at startup
# RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
#     TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('prithivida/Splade_PP_en_v1')"

# 3) application code
COPY taxlens ./taxlens

RUN mkdir -p /app/var && useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "taxlens.main:app", "--host", "0.0.0.0", "--port", "8000"]
