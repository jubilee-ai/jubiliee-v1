from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.shared.settings import get_settings

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context-manager for non-FastAPI code (Celery tasks, CLI scripts)."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (used for local dev; production uses Alembic)."""
    import logging

    from backend.shared.models import Base

    Base.metadata.create_all(bind=_get_engine())
    try:
        with get_db_session() as session:
            from backend.mrm.seed import seed_mrm_demo_data_if_needed

            seed_mrm_demo_data_if_needed(session)
    except Exception:
        logging.getLogger(__name__).warning(
            "MRM demo seed skipped (tables may not exist yet)",
            exc_info=True,
        )
