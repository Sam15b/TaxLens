"""FastAPI application factory.

Run locally:  uv run uvicorn taxlens.main:app --reload   ->  http://localhost:8000/docs
"""

from fastapi import FastAPI

from taxlens import __version__
from taxlens.api.errors import install_error_handlers
from taxlens.api.routes import ask, health
from taxlens.logging_config import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="TaxLens",
        version=__version__,
        description=(
            "Tax & legal document assistant for Indian tax law. "
            "Cited answers, document review and PII masking."
        ),
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(ask.router)
    # Week 3: auth, ingest, review, redact routers + tracing setup
    return app


app = create_app()
