"""Security unit tests for cryptographic artifact signing and integrity verification."""

import hashlib
import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier

from coffee_quality.security.integrity import (
    ModelIntegrityError,
    compute_artifact_sha256,
    compute_hmac_signature,
    safe_load_model,
    safe_save_model,
    sign_model_artifact,
    verify_model_artifact,
)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a temporary test file with known contents."""
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"coffee-lot-weights-v1")
    return file_path


def test_compute_artifact_sha256_success(sample_file: Path):
    """Verify streaming SHA-256 computation against hashlib reference."""
    expected_hash = hashlib.sha256(b"coffee-lot-weights-v1").hexdigest()
    actual_hash = compute_artifact_sha256(sample_file)
    assert actual_hash == expected_hash


def test_compute_artifact_sha256_errors(tmp_path: Path):
    """Verify error handling on non-existent files and directories."""
    missing_file = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError, match="Target file not found"):
        compute_artifact_sha256(missing_file)

    with pytest.raises(IsADirectoryError, match="Target path is a directory"):
        compute_artifact_sha256(tmp_path)


def test_compute_hmac_signature_success(sample_file: Path):
    """Verify HMAC-SHA256 signature calculation with string and bytes keys."""
    sig_str_key = compute_hmac_signature(sample_file, secret_key="super-secret")
    sig_bytes_key = compute_hmac_signature(sample_file, secret_key=b"super-secret")

    assert sig_str_key == sig_bytes_key
    assert len(sig_str_key) == 64


def test_compute_hmac_signature_errors(tmp_path: Path, sample_file: Path):
    """Verify error handling on missing files, directory paths, and empty keys."""
    missing_file = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        compute_hmac_signature(missing_file, secret_key="key")

    with pytest.raises(IsADirectoryError):
        compute_hmac_signature(tmp_path, secret_key="key")

    with pytest.raises(ValueError, match="HMAC secret key cannot be empty"):
        compute_hmac_signature(sample_file, secret_key="")

    with pytest.raises(ValueError, match="HMAC secret key cannot be empty"):
        compute_hmac_signature(sample_file, secret_key=b"")


def test_sign_and_verify_model_artifact_without_hmac(tmp_path: Path, sample_file: Path):
    """Verify artifact signing and verification lifecycle without HMAC."""
    manifest_path = tmp_path / "manifest.json"
    manifest = sign_model_artifact(
        artifact_path=sample_file,
        output_manifest_path=manifest_path,
        metadata={"version": "1.0.0"},
    )

    assert manifest["artifact_name"] == "model.bin"
    assert manifest["hmac_sha256"] is None
    assert manifest["metadata"]["version"] == "1.0.0"

    assert verify_model_artifact(sample_file, manifest_path) is True


def test_sign_and_verify_model_artifact_with_hmac(tmp_path: Path, sample_file: Path):
    """Verify artifact signing and verification lifecycle with HMAC secret key."""
    manifest_path = tmp_path / "manifest_hmac.json"
    secret_key = "prod-signing-secret"

    manifest = sign_model_artifact(
        artifact_path=sample_file,
        output_manifest_path=manifest_path,
        secret_key=secret_key,
        metadata={"environment": "production"},
    )

    assert manifest["hmac_sha256"] is not None
    assert verify_model_artifact(sample_file, manifest_path, secret_key=secret_key) is True


def test_tampered_artifact_raises_integrity_error(tmp_path: Path, sample_file: Path):
    """Verify that tampering with a single byte in the artifact raises ModelIntegrityError."""
    manifest_path = tmp_path / "manifest.json"
    sign_model_artifact(sample_file, manifest_path)

    # Tamper with file
    sample_file.write_bytes(b"tampered-malicious-bytes")

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch: artifact has been modified or corrupted"):
        verify_model_artifact(sample_file, manifest_path)


def test_tampered_manifest_hash_raises_integrity_error(tmp_path: Path, sample_file: Path):
    """Verify that a forged manifest with incorrect hash raises ModelIntegrityError."""
    manifest_path = tmp_path / "manifest.json"
    sign_model_artifact(sample_file, manifest_path)

    # Tamper with hash inside manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["sha256"] = "0" * 64
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        verify_model_artifact(sample_file, manifest_path)


def test_hmac_verification_failures(tmp_path: Path, sample_file: Path):
    """Verify errors when HMAC key is missing or signature mismatch occurs."""
    manifest_path = tmp_path / "manifest_hmac.json"
    sign_model_artifact(sample_file, manifest_path, secret_key="valid-secret")

    # Missing key when manifest expects HMAC
    with pytest.raises(ModelIntegrityError, match="Manifest contains an HMAC-SHA256 signature, but no secret_key"):
        verify_model_artifact(sample_file, manifest_path, secret_key=None)

    # Wrong secret key
    with pytest.raises(ModelIntegrityError, match="HMAC-SHA256 signature verification failed"):
        verify_model_artifact(sample_file, manifest_path, secret_key="wrong-secret")


def test_malformed_and_missing_manifest_errors(tmp_path: Path, sample_file: Path):
    """Verify handling of missing manifest files and corrupted JSON payloads."""
    missing_manifest = tmp_path / "missing_manifest.json"
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        verify_model_artifact(sample_file, missing_manifest)

    # Malformed JSON
    corrupted_manifest = tmp_path / "corrupted.json"
    corrupted_manifest.write_text("invalid-json{not_valid}", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="Malformed manifest JSON"):
        verify_model_artifact(sample_file, corrupted_manifest)

    # Missing required 'sha256' field
    incomplete_manifest = tmp_path / "incomplete.json"
    incomplete_manifest.write_text('{"other_field": 123}', encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="missing required 'sha256' field"):
        verify_model_artifact(sample_file, incomplete_manifest)


def test_safe_save_and_safe_load_lifecycle(tmp_path: Path):
    """Verify safe serialization and safe loading of a real estimator."""
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[1, 2], [3, 4]], [1, 1])

    artifact_path = tmp_path / "saved_model.bin"
    manifest_path = tmp_path / "saved_manifest.json"
    secret_key = "secure-signing-key"

    manifest = safe_save_model(
        model=model,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        secret_key=secret_key,
        metadata={"model_type": "DummyClassifier"},
    )

    assert manifest["sha256"] is not None

    loaded_model = safe_load_model(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        secret_key=secret_key,
    )

    assert isinstance(loaded_model, DummyClassifier)
    preds = loaded_model.predict([[5, 6]])
    assert preds[0] == 1


def test_safe_load_blocks_deserialization_on_tampering(tmp_path: Path):
    """Verify that tampering blocks deserialization before execution."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0], [1]], [1, 1])

    artifact_path = tmp_path / "model.bin"
    manifest_path = tmp_path / "manifest.json"

    safe_save_model(model, artifact_path, manifest_path)

    # Tamper with artifact binary
    artifact_path.write_bytes(b"tampered-malicious-payload")

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        safe_load_model(artifact_path, manifest_path)
