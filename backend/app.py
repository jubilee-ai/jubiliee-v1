"""
FastAPI composition root.
Domain handlers are organized by feature (catalog, training, chat).
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.shared.mlflow_setup import bootstrap_mlflow
from backend.shared.settings import bootstrap_paths, get_settings

bootstrap_paths()
bootstrap_mlflow()

from backend.catalog.routes import router as catalog_router
from backend.chat.routes import router as chat_router
from backend.experiments.routes import router as experiments_router
from backend.shared.database import init_db
from backend.training.routes import router as training_router


def _start_monitor_scheduler() -> None:
    if os.getenv("JUBILEE_MONITOR_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        from agents.subagents.monitor.agent import run_monitor_subagent
    except Exception as exc:
        print(f"[monitor] scheduler not started: {exc}")
        return

    ref = os.getenv("JUBILEE_MONITOR_REFERENCE_REF", "").strip()
    cur = os.getenv("JUBILEE_MONITOR_CURRENT_REF", "").strip()
    if not ref or not cur:
        print("[monitor] JUBILEE_MONITOR_ENABLED set but missing REF/CURRENT refs — skipping scheduler")
        return

    sched = BackgroundScheduler()

    def tick():
        try:
            run_monitor_subagent(
                {
                    "reference_dataset_ref": ref,
                    "current_dataset_ref": cur,
                    "prediction_column": os.getenv("JUBILEE_MONITOR_PRED_COL", ""),
                    "target_column": os.getenv("JUBILEE_MONITOR_TARGET_COL", ""),
                }
            )
        except Exception as e:
            print(f"[monitor] tick failed: {e}")

    hours = int(os.getenv("JUBILEE_MONITOR_INTERVAL_HOURS", "24") or "24")
    sched.add_job(tick, "interval", hours=max(1, hours), id="jubilee_monitor")
    sched.start()
    print(f"✅ Monitor scheduler started (every {hours}h)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Jubilee Training Agent API starting...")
    init_db()
    print("✅ Database tables verified")
    _start_monitor_scheduler()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
