"""
FastAPI composition root.
Domain handlers are organized by feature (catalog, training, chat).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.shared.settings import bootstrap_paths, get_settings

bootstrap_paths()

from backend.catalog.routes import router as catalog_router
from backend.chat.routes import router as chat_router
from backend.experiments.routes import router as experiments_router
from backend.notifications.routes import router as notifications_router
from backend.observability.routes import router as observability_router
from backend.observability.langfuse_proxy import router as langfuse_proxy_router
from backend.shared.database import init_db
from backend.training.routes import router as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Jubilee Training Agent API starting...")
    init_db()
    print("✅ Database tables verified")
    yield
    print("👋 Jubilee Training Agent API shutting down...")


settings = get_settings()
app = FastAPI(
    title="Jubilee Training Agent API",
    description="API for the ML Training Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "Jubilee Training Agent API"}


@app.get("/api/health")
async def health():
    db_ok = False
    try:
        from sqlalchemy import text
        from backend.shared.database import get_db_session
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "service": "Jubilee Training Agent API",
        "database": "connected" if db_ok else "unavailable",
    }


app.include_router(catalog_router)
app.include_router(training_router)
app.include_router(chat_router)
app.include_router(experiments_router)
app.include_router(notifications_router)
app.include_router(observability_router)
app.include_router(langfuse_proxy_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
