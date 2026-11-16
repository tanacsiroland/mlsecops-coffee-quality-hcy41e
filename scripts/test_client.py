"""Standalone Python client for exercising and validating the Coffee Quality API.

This script queries the REST API endpoints (/health, /predict, /monitoring/drift,
/monitoring/drift/html, and optionally /train) to verify service readiness and ML model inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api-client")

DEFAULT_SAMPLES: list[dict[str, Any]] = [
    {
        "Country.of.Origin": "Ethiopia",
        "Variety": "Yirgacheffe",
        "Processing.Method": "Washed / Wet",
        "Color": "Green",
        "altitude_mean_meters": 1950.0,
        "Moisture": 0.10,
        "Category.One.Defects": 0,
        "Category.Two.Defects": 1,
        "Quakers": 0,
        "Grading.Date": "2022-04-12",
    },
    {
        "Country.of.Origin": "Honduras",
        "Variety": "Bourbon",
        "Processing.Method": "Natural / Dry",
        "Color": "Brownish",
        "altitude_mean_meters": 850.0,
        "Moisture": 0.14,
        "Category.One.Defects": 4,
        "Category.Two.Defects": 12,
        "Quakers": 3,
        "Grading.Date": "2021-08-10",
    },
]


def run_health_check(client: httpx.Client, base_url: str) -> dict[str, Any]:
    """Query GET /health to inspect server readiness and active model metadata.

    Args:
        client: HTTP client instance.
        base_url: Base URL of the running API service.

    Returns:
        JSON response payload.
    """
    url = f"{base_url.rstrip('/')}/health"
    response = client.get(url, timeout=10.0)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    logger.info("Health Check Status: %s (Model Loaded: %s, Alias: %s)",
                data.get("status"), data.get("model_loaded"), data.get("model_alias"))
    return data


def run_predict(
    client: httpx.Client,
    base_url: str,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Query POST /predict with a batch of Arabica coffee lot samples.

    Args:
        client: HTTP client instance.
        base_url: Base URL of the running API service.
        samples: List of coffee feature dictionaries. Defaults to DEFAULT_SAMPLES.

    Returns:
        JSON response payload containing predictions and inference latency.
    """
    url = f"{base_url.rstrip('/')}/predict"
    items = samples if samples is not None else DEFAULT_SAMPLES
    payload = {"items": items}

    response = client.post(url, json=payload, timeout=15.0)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    predictions = data.get("predictions", [])
    logger.info("Inference completed: %d samples scored in %.2f ms (Model: %s)",
                len(predictions), data.get("inference_latency_ms", 0.0), data.get("model_version"))

    for idx, pred in enumerate(predictions):
        label = pred.get("classification_label", "Unknown")
        prob = pred.get("specialty_probability", 0.0)
        logger.info("  Sample #%d -> %s (Specialty Probability: %.4f)", idx + 1, label, prob)

    return data


def run_drift_check(client: httpx.Client, base_url: str, limit: int = 50) -> dict[str, Any]:
    """Query GET /monitoring/drift to evaluate data and target drift metrics.

    Args:
        client: HTTP client instance.
        base_url: Base URL of the running API service.
        limit: Maximum number of recent audit records to evaluate.

    Returns:
        JSON response payload with drift metrics.
    """
    url = f"{base_url.rstrip('/')}/monitoring/drift"
    response = client.get(url, params={"limit": limit}, timeout=30.0)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    drift_detected = data.get("dataset_drift", False)
    drift_share = data.get("share_of_drifted_columns", 0.0)
    ref_rows = data.get("reference_row_count", 0)
    curr_rows = data.get("current_row_count", 0)

    logger.info("Drift Evaluation: Drift Detected = %s (Drifted Share: %.1f%%, Ref Rows: %d, Live Rows: %d)",
                drift_detected, drift_share * 100.0, ref_rows, curr_rows)
    return data


def download_drift_html(
    client: httpx.Client,
    base_url: str,
    output_path: str | Path = "drift_report.html",
    limit: int = 50,
) -> Path:
    """Query GET /monitoring/drift/html and persist the interactive Evidently HTML report.

    Args:
        client: HTTP client instance.
        base_url: Base URL of the running API service.
        output_path: Destination file path for the HTML artifact.
        limit: Maximum number of recent audit records to evaluate.

    Returns:
        Resolved Path of the saved report.
    """
    url = f"{base_url.rstrip('/')}/monitoring/drift/html"
    response = client.get(url, params={"limit": limit}, timeout=30.0)
    response.raise_for_status()

    out_file = Path(output_path)
    out_file.write_text(response.text, encoding="utf-8")
    logger.info("Saved EvidentlyAI HTML drift report to: %s (%d bytes)", out_file.resolve(), len(response.text))
    return out_file


def run_retrain(
    client: httpx.Client,
    base_url: str,
    model_type: str = "lightgbm",
    promote: bool = True,
) -> dict[str, Any]:
    """Query POST /train to trigger full model pipeline retraining and promotion.

    Args:
        client: HTTP client instance.
        base_url: Base URL of the running API service.
        model_type: Model architecture to train ('lightgbm' or 'random_forest').
        promote: Whether to promote candidate to champion alias.

    Returns:
        JSON response payload with training metrics and run metadata.
    """
    url = f"{base_url.rstrip('/')}/train"
    payload = {
        "model_type": model_type,
        "promote_as_champion": promote,
    }
    logger.info("Triggering model retraining (%s)...", model_type)
    response = client.post(url, json=payload, timeout=120.0)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    metrics = data.get("metrics", {})
    logger.info("Retraining finished: Run ID = %s, Promoted = %s, F1 Specialty = %.4f, ROC-AUC = %.4f",
                data.get("run_id"), data.get("is_promoted"),
                metrics.get("f1_specialty", 0.0), metrics.get("roc_auc", 0.0))
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the verification client."""
    parser = argparse.ArgumentParser(
        description="Verify and exercise the Coffee Quality Classification REST API.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Also trigger POST /train to retrain and promote model pipeline",
    )
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "random_forest"],
        help="Model architecture for retraining (default: lightgbm)",
    )
    parser.add_argument(
        "--save-html",
        default=None,
        help="Optional path to save Evidently HTML drift report (e.g. drift_report.html)",
    )
    parser.add_argument(
        "--audit-limit",
        type=int,
        default=50,
        help="Number of audit records for drift evaluation (default: 50)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main client execution entrypoint."""
    args = parse_args(argv)

    print("=" * 70)
    print("  Arabica Coffee Quality API Verification Client")
    print(f"  Target: {args.base_url}")
    print("=" * 70)

    try:
        with httpx.Client() as client:
            # 1. Health check
            print("\n[1/4] Checking API Health and Model Readiness...")
            health_data = run_health_check(client, args.base_url)
            print(json.dumps(health_data, indent=2))

            # 2. Batch prediction
            print("\n[2/4] Executing Batch Coffee Lot Inference...")
            predict_data = run_predict(client, args.base_url)
            print(json.dumps(predict_data, indent=2))

            # 3. Drift summary & HTML report
            print("\n[3/4] Evaluating Live Data Drift...")
            drift_data = run_drift_check(client, args.base_url, limit=args.audit_limit)
            print(json.dumps(drift_data, indent=2))

            if args.save_html:
                download_drift_html(client, args.base_url, output_path=args.save_html, limit=args.audit_limit)

            # 4. Retraining (optional)
            if args.retrain:
                print(f"\n[4/4] Triggering Retraining Pipeline ({args.model_type})...")
                train_data = run_retrain(client, args.base_url, model_type=args.model_type)
                print(json.dumps(train_data, indent=2))
            else:
                print("\n[4/4] Retraining step skipped (pass --retrain to execute).")

            print("\n" + "=" * 70)
            print("  All API verification checks passed successfully!")
            print("=" * 70)
            return 0

    except httpx.ConnectError:
        logger.error(
            "Could not connect to API at %s. Ensure the service is running via:\n"
            "  uv run uvicorn coffee_quality.api.app:app --port 8000\n"
            "  or\n"
            "  docker compose up -d",
            args.base_url,
        )
        return 1
    except httpx.HTTPStatusError as exc:
        logger.error("HTTP request failed with status %d: %s", exc.response.status_code, exc.response.text)
        return 2
    except Exception as exc:
        logger.exception("Unexpected error during API verification: %s", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
