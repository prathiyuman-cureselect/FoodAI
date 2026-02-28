"""
NeuroVitals — Security, Encryption & Compliance
================================================
AES-256-GCM encryption, HMAC consent tokens, GDPR right-to-erasure,
and a simple RBAC decorator.
"""

import hashlib
import hmac
import json
import os
import shutil
import time
from functools import wraps
from typing import Callable, Optional, Set, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------------------------------------------------------------------------
# AES-256-GCM Encryption (HIPAA at-rest)
# ---------------------------------------------------------------------------

def generate_key() -> bytes:
    """Generate a random 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_blob(key: bytes, plaintext: bytes,
                 associated_data: bytes = b"") -> Tuple[bytes, bytes]:
    """Encrypt *plaintext* with AES-256-GCM.  Returns ``(nonce, ciphertext)``."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return nonce, ct


def decrypt_blob(key: bytes, nonce: bytes, ciphertext: bytes,
                 associated_data: bytes = b"") -> bytes:
    """Decrypt *ciphertext* previously encrypted with :func:`encrypt_blob`."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data)


# ---------------------------------------------------------------------------
# HMAC Consent Tokens (GDPR)
# ---------------------------------------------------------------------------

_CONSENT_SECRET = os.environ.get("NEUROVITALS_CONSENT_SECRET", "change-me-in-production")


def generate_consent_token(subject_id: str, scope: str = "analysis",
                           ttl_seconds: int = 86400,
                           secret: Optional[str] = None) -> str:
    """Create an HMAC-signed consent token.

    The token encodes the subject ID, permitted scope, and expiry
    timestamp as a base-64-style compact string.
    """
    secret = secret or _CONSENT_SECRET
    expiry = int(time.time()) + ttl_seconds
    payload = json.dumps({"sub": subject_id, "scope": scope, "exp": expiry},
                         separators=(",", ":"))
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    # token = payload_hex.signature
    return payload.encode().hex() + "." + sig


def verify_consent_token(token: str,
                         secret: Optional[str] = None) -> Optional[dict]:
    """Verify and decode a consent token.

    Returns the payload dict if valid and not expired, else ``None``.
    """
    secret = secret or _CONSENT_SECRET
    try:
        payload_hex, sig = token.rsplit(".", 1)
        payload_bytes = bytes.fromhex(payload_hex)
        expected_sig = hmac.new(secret.encode(), payload_bytes,
                                hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        payload = json.loads(payload_bytes)
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GDPR Right-to-Erasure
# ---------------------------------------------------------------------------

def erase_subject_data(subject_id: str, data_dir: str) -> int:
    """Delete all data for *subject_id* under *data_dir*.

    Searches for directories / files containing the subject_id in their
    name and removes them.  Returns the count of items removed.
    """
    removed = 0
    if not os.path.isdir(data_dir):
        return removed

    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if subject_id in name:
                os.remove(os.path.join(root, name))
                removed += 1
        for name in dirs:
            if subject_id in name:
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                removed += 1
    return removed


# ---------------------------------------------------------------------------
# Simple RBAC Decorator
# ---------------------------------------------------------------------------

_ROLE_PERMISSIONS: dict = {
    "admin":    {"read", "write", "delete", "analyze", "governance"},
    "clinician": {"read", "analyze"},
    "auditor":  {"read", "governance"},
    "patient":  {"read"},
}


def get_permissions(role: str) -> Set[str]:
    """Return the permission set for *role*."""
    return _ROLE_PERMISSIONS.get(role, set())


def require_role(*allowed_roles: str):
    """Decorator that checks the caller's role against allowed roles.

    The decorated function must accept a keyword argument ``role``.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            role = kwargs.get("role", "")
            if role not in allowed_roles:
                raise PermissionError(
                    f"Role '{role}' is not authorised. "
                    f"Required: {allowed_roles}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
