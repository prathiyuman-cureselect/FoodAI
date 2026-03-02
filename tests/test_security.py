"""Tests for NeuroVitals SDK — Security & Compliance."""
import os
import tempfile
import pytest

from sdk.security import (
    generate_key,
    encrypt_blob,
    decrypt_blob,
    generate_consent_token,
    verify_consent_token,
    erase_subject_data,
    get_permissions,
    require_role,
)


class TestAES256:
    def test_encrypt_decrypt_roundtrip(self):
        key = generate_key()
        plaintext = b"sensitive patient data"
        nonce, ct = encrypt_blob(key, plaintext)
        result = decrypt_blob(key, nonce, ct)
        assert result == plaintext

    def test_different_key_fails(self):
        key1 = generate_key()
        key2 = generate_key()
        nonce, ct = encrypt_blob(key1, b"data")
        with pytest.raises(Exception):
            decrypt_blob(key2, nonce, ct)

    def test_associated_data(self):
        key = generate_key()
        ad = b"subject-123"
        nonce, ct = encrypt_blob(key, b"data", associated_data=ad)
        result = decrypt_blob(key, nonce, ct, associated_data=ad)
        assert result == b"data"


class TestConsentTokens:
    def test_generate_and_verify(self):
        token = generate_consent_token("sub-001", scope="analysis",
                                       secret="test-secret")
        payload = verify_consent_token(token, secret="test-secret")
        assert payload is not None
        assert payload["sub"] == "sub-001"
        assert payload["scope"] == "analysis"

    def test_wrong_secret_rejects(self):
        token = generate_consent_token("sub-001", secret="secret-a")
        result = verify_consent_token(token, secret="secret-b")
        assert result is None

    def test_expired_token_rejected(self):
        token = generate_consent_token("sub-001", ttl_seconds=-1,
                                       secret="test")
        result = verify_consent_token(token, secret="test")
        assert result is None


class TestErasure:
    def test_erases_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files for a subject
            for fname in ["sub-001_data.json", "sub-001_video.mp4", "other.txt"]:
                with open(os.path.join(tmpdir, fname), "w") as f:
                    f.write("x")
            removed = erase_subject_data("sub-001", tmpdir)
            assert removed == 2
            assert os.path.exists(os.path.join(tmpdir, "other.txt"))


class TestRBAC:
    def test_admin_permissions(self):
        perms = get_permissions("admin")
        assert "analyze" in perms
        assert "governance" in perms

    def test_patient_permissions(self):
        perms = get_permissions("patient")
        assert "read" in perms
        assert "delete" not in perms

    def test_decorator_allows(self):
        @require_role("admin")
        def admin_fn(**kwargs):
            return True

        assert admin_fn(role="admin") is True

    def test_decorator_denies(self):
        @require_role("admin")
        def admin_fn(**kwargs):
            return True

        with pytest.raises(PermissionError):
            admin_fn(role="patient")
