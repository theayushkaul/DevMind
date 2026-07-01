"""
app/main.py
────────────
FastAPI application entry point.

Mounts the webhook router and configures logging. This is what Railway runs:
    uvicorn app.main:app --host 0.0.0.0 --port 8080

One app, one router, one endpoint — kept deliberately thin. All business
logic lives in the webhook/ and agent/ layers, not here.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from app.webhook.router import router as webhook_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DevMind",
    description="AI-powered GitHub PR review agent",
    version="0.1.0",
    # Disable docs in production — no need to expose the API schema publicly.
    docs_url=None if os.environ.get("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None,
)

app.include_router(webhook_router)


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint for Railway's uptime monitoring.
    Returns 200 as long as the process is alive.
    """
    return {"status": "ok"}


logger.info("DevMind webhook receiver started")
