#!/usr/bin/env bash
# ==============================================================================
# REST API Verification Script for Coffee Quality Classification Service
# ==============================================================================
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
echo "======================================================================"
echo "  Arabica Coffee Quality API Shell Verification"
echo "  Target URL: ${BASE_URL}"
echo "======================================================================"

# Check curl dependency
if ! command -v curl &>/dev/null; then
    echo "ERROR: 'curl' is required but not installed." >&2
    exit 1
fi

# Optional JSON pretty printing
format_json() {
    if command -v jq &>/dev/null; then
        jq .
    else
        cat
    fi
}

echo ""
echo "[1/4] Testing GET /health ..."
HEALTH_RESPONSE=$(curl -sS --fail-with-body -X GET "${BASE_URL}/health")
echo "${HEALTH_RESPONSE}" | format_json

echo ""
echo "[2/4] Testing POST /predict (batch inference) ..."
PREDICT_PAYLOAD='{
  "items": [
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
      "Grading.Date": "2022-04-12"
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
      "Grading.Date": "2021-08-10"
    }
  ]
}'

PREDICT_RESPONSE=$(curl -sS --fail-with-body -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${PREDICT_PAYLOAD}")
echo "${PREDICT_RESPONSE}" | format_json

echo ""
echo "[3/4] Testing GET /monitoring/drift ..."
DRIFT_RESPONSE=$(curl -sS --fail-with-body -X GET "${BASE_URL}/monitoring/drift?limit=20")
echo "${DRIFT_RESPONSE}" | format_json

echo ""
echo "[4/4] Testing GET /monitoring/drift/html (saving drift_report.html) ..."
curl -sS --fail-with-body -X GET "${BASE_URL}/monitoring/drift/html?limit=20" -o "drift_report.html"
echo "HTML report downloaded successfully to drift_report.html ($(wc -c < drift_report.html) bytes)."

echo ""
echo "======================================================================"
echo "  All shell verification tests succeeded!"
echo "======================================================================"
