"""Asmeranda backend entrypoint (FastAPI app)."""
from __future__ import annotations

# Setup imports for both Docker and local development
import import_helper
import_helper.setup_imports()

import logging
import sys
from pathlib import Path

# Agar ``import core.state`` dll bisa resolve ke project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.v1 import (
    datasets,
    eda,
    health,
    interpretation,
    preprocessing,
    timeseries,
    training,
    ws,
)

from core.config import settings


def create_app() -> FastAPI:
    """Factory function - memudahkan testing."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("asmeranda.backend")
    logger.info("Initializing %s v%s", settings.app_name, settings.app_version)

    # Initialize rate limiter
    limiter = Limiter(key_func=get_remote_address)
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        # Disable redundant docs URLs in production
        docs_url="/docs" if settings.debug else "/docs",
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Register rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS - allow all by default (development); restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(health.router, tags=["health"])
    app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
    app.include_router(eda.router, prefix="/api/v1/eda", tags=["eda"])
    app.include_router(
        preprocessing.router, prefix="/api/v1/preprocessing", tags=["preprocessing"]
    )
    app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
    
    # Note: Phase 1 endpoints are integrated into existing routers for immediate functionality:
    # - Clustering: Integrated into preprocessing router (/api/v1/preprocessing/cluster, /optimal-k)
    # - Optimization: Integrated into training router (/api/v1/training/optimize, /optimize-sync)
    # - Recommendations: Integrated into EDA router (/api/v1/eda/analyze)
    
    # Standalone routers for future use (disabled for now to avoid conflicts)
    # app.include_router(clustering.router, prefix="/api/v1/clustering", tags=["clustering"])
    # app.include_router(optimization.router, prefix="/api/v1/optimization", tags=["optimization"])
    # app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["recommendations"])
    
    app.include_router(
        interpretation.router,
        prefix="/api/v1/interpretation",
        tags=["interpretation"],
    )
    app.include_router(
        timeseries.router, prefix="/api/v1/timeseries", tags=["timeseries"]
    )
    app.include_router(ws.router, prefix="/api/v1/ws", tags=["websocket"])

    # Debug: Print all routes
    logger.info("Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path'):
            logger.info(f"  {route.path}")
        elif hasattr(route, 'prefix'):
            logger.info(f"  Router: {route.prefix}")

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
