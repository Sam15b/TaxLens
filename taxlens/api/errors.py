"""Error responses in the shared envelope: ``{"error": {"code", "message", "request_id"}}``."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from taxlens.api.schemas import ErrorBody, ErrorResponse
from taxlens.llm.providers import ProviderUnavailableError


def error_response(status: int, code: str, message: str, request: Request) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request.headers.get("x-request-id"))
    )
    return JSONResponse(status_code=status, content=body.model_dump())


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderUnavailableError)
    async def _llm_unavailable(request: Request, exc: ProviderUnavailableError) -> JSONResponse:
        return error_response(503, "llm_unavailable", str(exc), request)
