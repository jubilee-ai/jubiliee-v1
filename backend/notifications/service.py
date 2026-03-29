"""Notification creation, listing, and mark-read logic."""

from __future__ import annotations

import logging
from typing import Optional

from backend.shared.database import get_db_session
from backend.shared.models import Notification

logger = logging.getLogger(__name__)


def create_notification(
    *,
    type: str,
    title: str,
    body: Optional[str] = None,
    experiment_id: Optional[str] = None,
    job_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Insert a notification row. Safe to call from background threads."""
    try:
        with get_db_session() as session:
            n = Notification(
                type=type,
                title=title,
                body=body,
                experiment_id=experiment_id,
                job_id=job_id,
                metadata_=metadata,
                read=False,
            )
            session.add(n)
            session.commit()
    except Exception:
        logger.exception("Failed to create notification")


def list_recent(limit: int = 20, unread_only: bool = False) -> list[dict]:
    """Return recent notifications, newest first."""
    with get_db_session() as session:
        q = session.query(Notification)
        if unread_only:
            q = q.filter(Notification.read == False)  # noqa: E712
        rows = q.order_by(Notification.created_at.desc()).limit(limit).all()
        return [
            {
                "id": str(r.id),
                "experiment_id": r.experiment_id,
                "job_id": r.job_id,
                "type": r.type,
                "title": r.title,
                "body": r.body,
                "metadata": r.metadata_,
                "read": r.read,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def mark_read(notification_ids: list[str]) -> int:
    """Mark notifications as read. Returns count updated."""
    import uuid

    uuids = []
    for nid in notification_ids:
        try:
            uuids.append(uuid.UUID(nid))
        except (ValueError, AttributeError):
            continue
    if not uuids:
        return 0
    with get_db_session() as session:
        count = (
            session.query(Notification)
            .filter(Notification.id.in_(uuids))
            .update({"read": True}, synchronize_session=False)
        )
        session.commit()
        return count
