"""
Clerk JWT authentication dependencies for FastAPI (PyJWT + JWKS).

Set ``CLERK_JWKS_URL`` in the environment (Clerk Dashboard → API Keys → JWT / JWKS).

Usage in route handlers:
    from backend.shared.auth import ClerkUser, require_auth, require_admin

    @router.get("/api/things")
    def list_things(user: ClerkUser = Depends(require_auth)):
        # user.user_id, user.org_id, user.org_role available
        ...

    @router.delete("/api/things/{id}")
    def delete_thing(id: str, user: ClerkUser = Depends(require_admin)):
        # only org:admin can reach here
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jwt
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.algorithms import RSAAlgorithm

from backend.shared.settings import get_settings

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    """Fetch and cache the JWKS key set from Clerk."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    settings = get_settings()
    jwks_url = settings.CLERK_JWKS_URL
    if not jwks_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CLERK_JWKS_URL is not configured",
        )

    resp = requests.get(jwks_url, timeout=10)
    resp.raise_for_status()
    _jwks_cache = resp.json()
    return _jwks_cache


def _decode_token(token: str) -> dict:
    """Validate a Clerk JWT and return its claims."""
    jwks = _get_jwks()
    keys = jwks.get("keys", [])
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No keys found in Clerk JWKS",
        )

    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    matching_key = None
    for key in keys:
        if key.get("kid") == kid:
            matching_key = key
            break

    if matching_key is None:
        # JWKS may have rotated; clear cache and retry once.
        global _jwks_cache
        _jwks_cache = None
        jwks = _get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                matching_key = key
                break

    if matching_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token signing key not found in JWKS",
        )

    public_key = RSAAlgorithm.from_jwk(matching_key)
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False, "leeway": 5},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )

    return payload


def _coerce_org_claim(raw: object) -> Optional[str]:
    """Session templates may put a plain org id string or an org object in JWT claims."""
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        return s or None
    if isinstance(raw, dict):
        for key in ("id", "org_id", "organization_id"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    return None


@dataclass
class ClerkUser:
    user_id: str
    org_id: Optional[str]
    org_role: Optional[str]


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> ClerkUser:
    """Validate the Clerk JWT and return the authenticated user context."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    payload = _decode_token(credentials.credentials)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    # Clerk session claims: add org_id / org_role in Dashboard → Sessions → Customize session token,
    # or use short keys like "o" depending on template. org_id may be a string or an embedded object.
    org_id = _coerce_org_claim(payload.get("org_id")) or _coerce_org_claim(
        payload.get("o")
    )
    raw_role = payload.get("org_role") or payload.get("org:role")
    if isinstance(raw_role, dict):
        org_role = (
            (raw_role.get("role") or raw_role.get("r") or raw_role.get("rol"))
        )
        org_role = str(org_role).strip() if org_role else None
    else:
        org_role = str(raw_role).strip() if raw_role else None

    # Custom session templates sometimes use short role slugs (e.g. "admin" in a nested object).
    if org_role == "admin":
        org_role = "org:admin"
    elif org_role == "member":
        org_role = "org:member"

    return ClerkUser(
        user_id=str(user_id),
        org_id=org_id,
        org_role=org_role,
    )


async def require_org(
    user: ClerkUser = Depends(require_auth),
) -> ClerkUser:
    """Require an active organization (org claims present on the session JWT)."""
    if not user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Active organization required. Select an organization in the app header.",
        )
    return user


async def require_admin(
    user: ClerkUser = Depends(require_auth),
) -> ClerkUser:
    """Require the authenticated user to have the org:admin role."""
    if user.org_role != "org:admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
