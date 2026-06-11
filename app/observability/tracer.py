import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    return _trace_id_var.get()

class TracingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, service_name: str = "company-research") -> None:
        super().__init__(app)
        self.service_name = service_name

    async def dispatch(
            self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
        token = _trace_id_var.set(trace_id)

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "service=%s trace_id=%s method=%s path=%s duration_ms=%.1f status=500",
                self.service_name,
                trace_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        finally:
            _trace_id_var.reset(token)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "service=%s trace_id=%s method=%s path=%s status=%d duration_ms=%.1f",
            self.service_name,
            trace_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Trace-Id"] = trace_id
        return response


def add_tracing(app: FastAPI, service_name: str = "company-research") -> None:
    app.add_middleware(TracingMiddleware, service_name=service_name)
    logger.debug("tracer: TracingMiddleware registered for service=%s", service_name)
