"""Streamlit web interface for Arabica coffee quality prediction and MLSecOps monitoring."""

from __future__ import annotations

from datetime import date
import io
import logging
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"


def get_api_url() -> str:
    """Retrieve API base URL from environment or fallback default."""
    url = os.getenv("COFFEE_API_URL") or os.getenv("API_URL") or DEFAULT_API_URL
    return url.rstrip("/")


def fetch_service_health(api_url: str | None = None) -> dict[str, Any] | None:
    """Fetch health and model registry status from the backend serving API.

    Args:
        api_url: Base serving URL. Defaults to configured environment URL.

    Returns:
        Dictionary of health response data, or None if connection fails.
    """
    base = api_url or get_api_url()
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base}/health")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data
            return None
    except Exception as exc:
        logger.debug("Health check failed for %s: %s", base, exc)
        return None


def predict_single_lot(
    payload: dict[str, Any],
    api_url: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Submit a single coffee lot's attributes to the /predict API endpoint.

    Args:
        payload: Dictionary of sample attributes.
        api_url: Base serving URL.

    Returns:
        Tuple of (prediction_item, error_message).
    """
    base = api_url or get_api_url()
    body = {"items": [payload]}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{base}/predict", json=body)
            if resp.status_code == 200:
                data = resp.json()
                preds = data.get("predictions", [])
                if preds:
                    return preds[0], None
                return None, "Empty prediction response from serving service."
            return None, f"Serving service error {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, f"Connection failed to {base}: {exc}"


def predict_csv_batch(
    csv_df: pd.DataFrame,
    api_url: str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Process a batch of coffee records from a DataFrame through /predict.

    Args:
        csv_df: Tabular input DataFrame with coffee lot attributes.
        api_url: Base serving URL.

    Returns:
        Tuple of (annotated_DataFrame, error_message).
    """
    base = api_url or get_api_url()
    if csv_df.empty:
        return None, "Uploaded CSV file contains no records."

    records = csv_df.to_dict(orient="records")
    body = {"items": records}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base}/predict", json=body)
            if resp.status_code == 200:
                data = resp.json()
                predictions = data.get("predictions", [])
                annotated = csv_df.copy()
                annotated["Predicted_Is_Specialty"] = [p.get("is_specialty") for p in predictions]
                annotated["Specialty_Probability"] = [p.get("probability") for p in predictions]
                annotated["Confidence"] = [p.get("confidence") for p in predictions]
                annotated["Classification"] = [p.get("label") for p in predictions]
                return annotated, None
            return None, f"Prediction error {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, f"Batch inference failed: {exc}"


def fetch_drift_summary(
    limit: int = 500,
    api_url: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Retrieve statistical data and target drift summary from /monitoring/drift.

    Args:
        limit: Number of recent inference audit logs to evaluate.
        api_url: Base serving URL.

    Returns:
        Tuple of (summary_dict, error_message).
    """
    base = api_url or get_api_url()
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{base}/monitoring/drift", params={"limit": limit})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data, None
            return None, f"Drift summary error {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, f"Failed retrieving drift summary: {exc}"


def fetch_drift_html(
    limit: int = 500,
    api_url: str | None = None,
) -> tuple[str | None, str | None]:
    """Retrieve rendered EvidentlyAI interactive HTML report from /monitoring/drift/html.

    Args:
        limit: Number of recent inference audit logs to evaluate.
        api_url: Base serving URL.

    Returns:
        Tuple of (html_content, error_message).
    """
    base = api_url or get_api_url()
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(f"{base}/monitoring/drift/html", params={"limit": limit})
            if resp.status_code == 200:
                return resp.text, None
            return None, f"Drift report error {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, f"Failed retrieving drift HTML: {exc}"


def trigger_pipeline_retraining(
    model_type: str = "random_forest",
    register_as_champion: bool = True,
    api_url: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Trigger model retraining pipeline through /train endpoint.

    Args:
        model_type: Classifier architecture ('random_forest' or 'lightgbm').
        register_as_champion: Whether to promote candidate model to 'champion'.
        api_url: Base serving URL.

    Returns:
        Tuple of (retraining_response_dict, error_message).
    """
    base = api_url or get_api_url()
    payload = {
        "model_type": model_type,
        "register_as_champion": register_as_champion,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{base}/train", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data, None
            elif resp.status_code == 409:
                return None, "Retraining job already in progress. Please wait."
            return None, f"Retraining error {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, f"Retraining trigger failed: {exc}"


def render_app() -> None:
    """Render the full interactive Streamlit multi-tab dashboard."""
    st.set_page_config(
        page_title="Arabica Coffee Quality MLSecOps",
        page_icon="☕",
        layout="wide",
    )

    st.title("☕ Arabica Coffee Specialty Classification & MLSecOps Platform")
    st.markdown(
        "Secure, reproducible classification and drift monitoring for CQI Arabica coffee quality."
    )

    api_url = get_api_url()

    # Sidebar: System Health & Status
    with st.sidebar:
        st.header("Service Status")
        health = fetch_service_health(api_url)
        if health:
            st.success("API: Operational")
            st.metric("Environment", health.get("app_env", "Unknown"))
            model_loaded = health.get("model_loaded", False)
            if model_loaded:
                st.info(f"Model Alias: {health.get('model_alias', 'None')}")
            else:
                st.warning("Model: Not loaded")
        else:
            st.error("API: Offline / Unreachable")
            st.caption(f"Connecting to: `{api_url}`")

        nav_choice = st.radio(
            "Navigation",
            [
                "☕ Single Lot Scoring",
                "📊 Batch CSV Upload",
                "📈 Evidently Drift Dashboard",
                "⚙️ Model Retraining",
            ],
        )

    # 1. Single Lot Evaluation View
    if nav_choice == "☕ Single Lot Scoring":
        st.subheader("Evaluate Individual Coffee Lot")
        st.markdown("Enter lot characteristics to predict Specialty Grade classification status.")

        with st.form("single_eval_form"):
            col1, col2 = st.columns(2)
            with col1:
                country = st.selectbox(
                    "Country of Origin",
                    ["Ethiopia", "Colombia", "Guatemala", "Brazil", "Kenya", "Mexico", "Honduras", "Other"],
                    index=0,
                )
                variety = st.selectbox(
                    "Variety",
                    ["Typica", "Bourbon", "Caturra", "SL28", "Geisha", "Yellow Bourbon", "Other"],
                    index=0,
                )
                method = st.selectbox(
                    "Processing Method",
                    ["Washed / Wet", "Natural / Dry", "Semi-Washed / Honey", "Other"],
                    index=0,
                )
                color = st.selectbox(
                    "Bean Color",
                    ["Green", "Bluish-Green", "Pale Yellow", "Yellow-Green", "None"],
                    index=0,
                )
            with col2:
                altitude = st.slider("Altitude Mean (meters)", min_value=500.0, max_value=3500.0, value=1800.0)
                moisture = st.slider("Moisture Content", min_value=0.05, max_value=0.20, value=0.11, step=0.005)
                cat1_defects = st.number_input("Category One Defects (Primary)", min_value=0, max_value=20, value=0)
                cat2_defects = st.number_input("Category Two Defects (Secondary)", min_value=0, max_value=30, value=1)
                quakers = st.number_input("Quakers Count", min_value=0, max_value=10, value=0)
                grading_date = st.date_input("Grading Date", value=date(2021, 1, 1))

            submitted = st.form_submit_button("Grade Coffee Lot", type="primary")

        if submitted:
            payload = {
                "country_of_origin": country,
                "variety": variety,
                "processing_method": method,
                "color": color,
                "altitude_mean_meters": float(altitude),
                "moisture": float(moisture),
                "category_one_defects": int(cat1_defects),
                "category_two_defects": int(cat2_defects),
                "quakers": int(quakers),
                "grading_date": str(grading_date),
            }

            with st.spinner("Scoring lot via serving API..."):
                result, error = predict_single_lot(payload, api_url=api_url)

            if error:
                st.error(error)
            elif result:
                is_specialty = result.get("is_specialty", 0)
                prob = result.get("probability", 0.0)
                conf = result.get("confidence", 0.0)

                res_col1, res_col2, res_col3 = st.columns(3)
                if is_specialty == 1:
                    st.success("### 🌟 Specialty Grade Arabica (Score >= 80.0)")
                else:
                    st.warning("### 📦 Commercial Grade Arabica (Score < 80.0)")

                with res_col1:
                    st.metric("Classification", result.get("label", "Unknown"))
                with res_col2:
                    st.metric("Specialty Probability", f"{prob * 100:.1f}%")
                with res_col3:
                    st.metric("Model Confidence", f"{conf * 100:.1f}%")

    # 2. Batch CSV Upload View
    elif nav_choice == "📊 Batch CSV Upload":
        st.subheader("Batch Coffee Quality Scoring")
        st.markdown("Upload a CSV file containing multiple Arabica lot records for bulk classification.")

        uploaded_file = st.file_uploader("Upload Coffee CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                csv_df = pd.read_csv(uploaded_file)
                st.write(f"Previewing first 5 rows (Total: {len(csv_df)} lots):")
                st.dataframe(csv_df.head(5))

                if st.button("Score Batch Lots", type="primary"):
                    with st.spinner("Executing batch inference..."):
                        annotated_df, err = predict_csv_batch(csv_df, api_url=api_url)

                    if err:
                        st.error(err)
                    elif annotated_df is not None:
                        st.success(f"Successfully scored {len(annotated_df)} coffee samples.")
                        spec_count = int((annotated_df["Predicted_Is_Specialty"] == 1).sum())
                        comm_count = len(annotated_df) - spec_count
                        pct_spec = (spec_count / len(annotated_df)) * 100.0

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Lots", len(annotated_df))
                        m2.metric("Specialty Grade", f"{spec_count} ({pct_spec:.1f}%)")
                        m3.metric("Commercial Grade", comm_count)

                        st.dataframe(annotated_df)

                        buffer = io.StringIO()
                        annotated_df.to_csv(buffer, index=False)
                        st.download_button(
                            label="Download Annotated CSV",
                            data=buffer.getvalue(),
                            file_name="coffee_quality_predictions.csv",
                            mime="text/csv",
                        )
            except Exception as read_err:
                st.error(f"Failed to read CSV: {read_err}")

    # 3. EvidentlyAI Drift Dashboard View
    elif nav_choice == "📈 Evidently Drift Dashboard":
        st.subheader("EvidentlyAI Data & Concept Drift Monitoring")
        st.markdown(
            "Monitors statistical distribution shifts by comparing active production inferences "
            "against the validated baseline training dataset."
        )

        limit = st.slider("Audit Samples to Analyze", min_value=10, max_value=2000, value=500, step=50)

        if st.button("Refresh Drift Analysis", type="primary"):
            with st.spinner("Computing statistical drift metrics..."):
                summary, summary_err = fetch_drift_summary(limit=limit, api_url=api_url)
                html_report, html_err = fetch_drift_html(limit=limit, api_url=api_url)

            if summary_err:
                st.error(summary_err)
            elif summary:
                if summary.get("status") == "insufficient_samples":
                    st.info(
                        f"Insufficient live audit samples: {summary.get('current_sample_count', 0)} provided. "
                        "At least 5 samples are required for statistical testing."
                    )
                else:
                    drift_detected = summary.get("dataset_drift_detected", False)
                    share = summary.get("share_of_drifted_features", 0.0)
                    drifted_cnt = summary.get("drifted_features_count", 0)
                    total_cnt = summary.get("total_features_count", 0)

                    c1, c2, c3 = st.columns(3)
                    if drift_detected:
                        c1.metric("Dataset Drift", "DETECTED", delta="ALERT", delta_color="inverse")
                    else:
                        c1.metric("Dataset Drift", "STABLE", delta="OK")

                    c2.metric("Drifted Features", f"{drifted_cnt} / {total_cnt}")
                    c3.metric("Drift Share", f"{share * 100:.1f}%")

                    # Feature drift table
                    features = summary.get("drift_by_feature", {})
                    if features:
                        feature_rows = [
                            {
                                "Feature": k,
                                "Test Method": v.get("method"),
                                "Score / P-Value": v.get("score"),
                                "Threshold": v.get("threshold"),
                                "Drift Detected": "DRIFT" if v.get("drift_detected") else "Stable",
                            }
                            for k, v in features.items()
                        ]
                        st.table(pd.DataFrame(feature_rows))

                if html_report:
                    st.markdown("### Interactive EvidentlyAI Report")
                    components.html(html_report, height=900, scrolling=True)

    # 4. Model Retraining View
    elif nav_choice == "⚙️ Model Retraining":
        st.subheader("Trigger Automated Model Retraining Pipeline")
        st.markdown(
            "Execute end-to-end retraining, cross-validation, and MLflow champion model promotion."
        )

        with st.form("retrain_form"):
            model_type = st.selectbox(
                "Model Architecture",
                ["random_forest", "lightgbm"],
                index=0,
            )
            promote = st.checkbox("Promote model to 'champion' alias upon completion", value=True)
            submitted_train = st.form_submit_button("Start Retraining Pipeline", type="primary")

        if submitted_train:
            with st.spinner("Retraining pipeline executing in background (5-fold CV & holdout eval)..."):
                train_data, train_err = trigger_pipeline_retraining(
                    model_type=model_type,
                    register_as_champion=promote,
                    api_url=api_url,
                )

            if train_err:
                st.error(train_err)
            elif train_data:
                st.success("Retraining pipeline completed successfully!")
                t1, t2, t3 = st.columns(3)
                t1.metric("Run ID", str(train_data.get("run_id", "N/A"))[:12])
                metrics = train_data.get("metrics", {})
                t2.metric("Holdout Macro-F1", f"{metrics.get('macro_f1', 0.0):.4f}")
                t3.metric("Holdout ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")


if __name__ == "__main__":
    render_app()
