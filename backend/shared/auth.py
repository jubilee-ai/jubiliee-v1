"""
Clerk JWT verification and user resolution for FastAPI.

Flow:
  1. Frontend sends ``Authorization: Bearer <clerk-session-token>``
  2. This module verifies the JWT against Clerk's JWKS endpoint
  3. Looks up (or auto-creates) the internal User row
  4. Returns a lightweight ``CurrentUser`` context for route handlers
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Request

from backend.shared.settings import get_settings

log = logging.getLogger(__name__)

DEFAULT_ORG_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

_jwk_client: PyJWKClient | None = None


@dataclass(frozen=True, slots=True)
class CurrentUser:
    id: uuid.UUID
    clerk_id: str
    organization_id: uuid.UUID


def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        settings = get_settings()
        jwks_url = f"{settings.CLERK_ISSUER.rstrip('/')}/.well-known/jwks.json"
        _jwk_client = PyJWKClient(jwks_url, cache_keys=True)
    return _jwk_client


def _verify_token(token: str) -> dict:
    """Verify a Clerk JWT and return its claims."""
    settings = get_settings()
    try:
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
            issuer=settings.CLERK_ISSUER,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except Exception as exc:
        log.debug("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid authentication token")


def _resolve_user(clerk_id: str, claims: dict) -> CurrentUser:
    """Look up or auto-create the internal user for a Clerk ID."""
    from backend.shared.database import get_db_session
    from backend.shared.models import Organization, User

    with get_db_session() as session:
        user = session.query(User).filter_by(clerk_id=clerk_id).first()

        if user is None:
            org = session.query(Organization).filter_by(id=DEFAULT_ORG_ID).first()
            if org is None:
                org = Organization(id=DEFAULT_ORG_ID, name="Default Organization")
                session.add(org)
                session.flush()

            user = User(
                clerk_id=clerk_id,
                email=claims.get("email"),
                name=claims.get("name"),
                organization_id=org.id,
            )
            session.add(user)
            session.flush()
            log.info("Auto-created user %s (clerk=%s)", user.id, clerk_id)

        return CurrentUser(
            id=user.id,
            clerk_id=clerk_id,
            organization_id=user.organization_id,
        )


async def get_current_user(request: Request) -> CurrentUser:
    """FastAPI dependency — extracts and verifies the Clerk JWT, resolves internal user."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authentication token")

    token = auth_header.split(" ", 1)[1]
    claims = _verify_token(token)

    clerk_id = claims.get("sub")
    if not clerk_id:
        raise HTTPException(status_code=401, detail="Token missing subject claim")

    return _resolve_user(clerk_id, claims)
