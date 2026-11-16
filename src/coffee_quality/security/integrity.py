"""Cryptographic artifact signing, SHA-256 integrity verification, and safe deserialization."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import cloudpickle


class ModelIntegrityError(Exception):
    """Raised when artifact cryptographic verification fails or tampering is detected."""


def compute_artifact_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hexadecimal digest of a file using streaming chunks.

    Args:
        file_path: Filesystem path to the target file.
        chunk_size: Read buffer size in bytes (default 64 KB).

    Returns:
        Lowercase hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the target file does not exist.
        IsADirectoryError: If the path is a directory.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Target file not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Target path is a directory, not a file: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_hmac_signature(
    file_path: Path,
    secret_key: str | bytes,
    chunk_size: int = 65536,
) -> str:
    """Compute an HMAC-SHA256 signature for a file using a cryptographic secret key.

    Args:
        file_path: Filesystem path to the target file.
        secret_key: Secret key string or raw bytes for signing.
        chunk_size: Read buffer size in bytes (default 64 KB).

    Returns:
        Lowercase hexadecimal HMAC-SHA256 signature string.

    Raises:
        FileNotFoundError: If the target file does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If the secret key is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Target file not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Target path is a directory, not a file: {file_path}")

    key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    if not key_bytes:
        raise ValueError("HMAC secret key cannot be empty.")

    signer = hmac.new(key_bytes, digestmod=hashlib.sha256)
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            signer.update(chunk)
    return signer.hexdigest().lower()


def sign_model_artifact(
    artifact_path: Path,
    output_manifest_path: Path,
    secret_key: str | bytes | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and write a cryptographically signed JSON manifest for a model artifact.

    Args:
        artifact_path: Path to the serialized model file.
        output_manifest_path: Path where the signed manifest JSON will be written.
        secret_key: Optional secret key to generate an HMAC-SHA256 signature.
        metadata: Optional dictionary of domain or model metadata (e.g. metrics, version).

    Returns:
        The manifest dictionary that was written.
    """
    sha256_hash = compute_artifact_sha256(artifact_path)
    hmac_signature = compute_hmac_signature(artifact_path, secret_key) if secret_key else None

    manifest: dict[str, Any] = {
        "artifact_name": artifact_path.name,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "sha256": sha256_hash,
        "hmac_sha256": hmac_signature,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_model_artifact(
    artifact_path: Path,
    manifest_path: Path,
    secret_key: str | bytes | None = None,
) -> bool:
    """Verify that a model artifact matches its manifest hash and HMAC signature.

    Uses constant-time comparison (hmac.compare_digest) to prevent timing side-channel attacks.

    Args:
        artifact_path: Path to the model artifact file.
        manifest_path: Path to the signed JSON manifest file.
        secret_key: Optional secret key for HMAC verification. Required if manifest has hmac_sha256.

    Returns:
        True if all cryptographic checks pass.

    Raises:
        FileNotFoundError: If the artifact or manifest file is missing.
        ModelIntegrityError: If the manifest is malformed, hash mismatch occurs,
            or HMAC signature verification fails.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ModelIntegrityError(f"Malformed manifest JSON: {exc}") from exc

    if not isinstance(manifest, dict) or "sha256" not in manifest:
        raise ModelIntegrityError("Manifest is missing required 'sha256' field.")

    expected_sha256 = str(manifest["sha256"]).lower()
    actual_sha256 = compute_artifact_sha256(artifact_path)

    # Constant-time comparison
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ModelIntegrityError(
            f"SHA-256 mismatch: artifact has been modified or corrupted. "
            f"Expected: {expected_sha256}, Got: {actual_sha256}"
        )

    expected_hmac = manifest.get("hmac_sha256")
    if expected_hmac is not None:
        if secret_key is None:
            raise ModelIntegrityError(
                "Manifest contains an HMAC-SHA256 signature, but no secret_key was provided for verification."
            )
        actual_hmac = compute_hmac_signature(artifact_path, secret_key)
        if not hmac.compare_digest(actual_hmac, str(expected_hmac).lower()):
            raise ModelIntegrityError("HMAC-SHA256 signature verification failed: invalid key or tampered artifact.")

    return True


def safe_save_model(
    model: Any,
    artifact_path: Path,
    manifest_path: Path,
    secret_key: str | bytes | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a model to disk and generate a signed cryptographic manifest.

    Args:
        model: Python object/estimator to serialize.
        artifact_path: Destination path for serialized binary.
        manifest_path: Destination path for signed manifest JSON.
        secret_key: Optional secret key for HMAC signature.
        metadata: Optional metadata dictionary to record in manifest.

    Returns:
        The written manifest dictionary.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        cloudpickle.dump(model, f)

    return sign_model_artifact(
        artifact_path=artifact_path,
        output_manifest_path=manifest_path,
        secret_key=secret_key,
        metadata=metadata,
    )


def safe_load_model(
    artifact_path: Path,
    manifest_path: Path,
    secret_key: str | bytes | None = None,
) -> Any:
    """Verify model cryptographic integrity and deserialize only if verification succeeds.

    Args:
        artifact_path: Path to serialized model file.
        manifest_path: Path to signed manifest JSON file.
        secret_key: Optional secret key for HMAC verification.

    Returns:
        The deserialized model object.

    Raises:
        ModelIntegrityError: If verification fails (deserialization is blocked).
        FileNotFoundError: If the artifact or manifest file does not exist.
    """
    verify_model_artifact(artifact_path, manifest_path, secret_key=secret_key)
    with open(artifact_path, "rb") as f:
        return cloudpickle.load(f)
