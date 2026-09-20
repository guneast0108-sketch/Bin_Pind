import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import (
    CostLimitExceeded,
    PlaceNotFound,
    VideoNotFound,
    WebhookAuthError,
)
from app.settings import settings

log = structlog.get_logger()

app = FastAPI(
    title="Pind API",
    version="0.1.0",
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(VideoNotFound)
async def video_not_found_handler(_: Request, exc: VideoNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(PlaceNotFound)
async def place_not_found_handler(_: Request, exc: PlaceNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(CostLimitExceeded)
async def cost_limit_handler(_: Request, exc: CostLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=402, content={"detail": str(exc)})


@app.exception_handler(WebhookAuthError)
async def webhook_auth_handler(_: Request, exc: WebhookAuthError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


# ── Routers ───────────────────────────────────────────────────────────────────
# routers will be registered here as they are built in Phase 1+


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
