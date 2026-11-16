"""Unit tests for Pydantic API request and response schemas."""

import pandas as pd
from pydantic import ValidationError
import pytest

from coffee_quality.api.schemas import (
    CoffeeBatchRequest,
    CoffeePredictionInput,
    PredictionItem,
    PredictionResponse,
    TrainingTriggerRequest,
    TrainingTriggerResponse,
)


def test_coffee_prediction_input_defaults_and_aliases():
    """Verify default values and alias resolution."""
    input_item = CoffeePredictionInput()
    assert input_item.country_of_origin == "Unknown"
    assert input_item.altitude_mean_meters == 1310.6
    assert input_item.moisture == 0.10

    # Test alias population
    aliased_item = CoffeePredictionInput(
        **{
            "Country.of.Origin": "Guatemala",
            "Variety": "Bourbon",
            "Processing.Method": "Washed",
            "Color": "Green",
            "altitude_mean_meters": 1750.0,
            "Moisture": 0.11,
            "Category.One.Defects": 1,
            "Category.Two.Defects": 2,
            "Quakers": 0,
            "Grading.Date": "2021-05-15",
        }
    )
    assert aliased_item.country_of_origin == "Guatemala"
    assert aliased_item.variety == "Bourbon"
    assert aliased_item.altitude_mean_meters == 1750.0


def test_coffee_prediction_input_validation_errors():
    """Verify boundary checks on altitude, moisture, and defect counts."""
    # Altitude out of bounds (> 4000)
    with pytest.raises(ValidationError, match="altitude_mean_meters"):
        CoffeePredictionInput(altitude_mean_meters=4500.0)

    # Altitude negative
    with pytest.raises(ValidationError, match="altitude_mean_meters"):
        CoffeePredictionInput(altitude_mean_meters=-10.0)

    # Moisture ratio > 1.0
    with pytest.raises(ValidationError, match="moisture"):
        CoffeePredictionInput(moisture=1.5)

    # Negative defects
    with pytest.raises(ValidationError, match="category_one_defects"):
        CoffeePredictionInput(category_one_defects=-1)


def test_coffee_prediction_input_extra_forbidden_anti_leakage():
    """Verify that sensory scores and targets are rejected with ValidationError."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CoffeePredictionInput.model_validate({"Total.Cup.Points": 84.5})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        CoffeePredictionInput.model_validate({"Is_Specialty": 1})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        CoffeePredictionInput.model_validate({"Aroma": 8.0})


def test_coffee_batch_request_to_dataframe():
    """Verify batch request conversion to a DataFrame matching feature columns."""
    sample_1 = CoffeePredictionInput(country_of_origin="Ethiopia", altitude_mean_meters=1900.0)
    sample_2 = CoffeePredictionInput(country_of_origin="Colombia", altitude_mean_meters=1600.0)

    batch = CoffeeBatchRequest(items=[sample_1, sample_2])
    df = batch.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected_cols = [
        "Country.of.Origin",
        "Variety",
        "Processing.Method",
        "Color",
        "altitude_mean_meters",
        "Moisture",
        "Category.One.Defects",
        "Category.Two.Defects",
        "Quakers",
        "Grading.Date",
    ]
    assert list(df.columns) == expected_cols
    assert df.loc[0, "Country.of.Origin"] == "Ethiopia"
    assert df.loc[1, "Country.of.Origin"] == "Colombia"


def test_coffee_batch_request_bounds():
    """Verify min_length=1 and max_length=500 constraints."""
    # Empty items list
    with pytest.raises(ValidationError, match="too_short"):
        CoffeeBatchRequest(items=[])

    # Oversized items list (> 500)
    oversized = [CoffeePredictionInput() for _ in range(501)]
    with pytest.raises(ValidationError, match="too_long"):
        CoffeeBatchRequest(items=oversized)

    # Extra attributes forbidden
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CoffeeBatchRequest.model_validate({
            "items": [CoffeePredictionInput().model_dump()],
            "unauthorized_field": "injected",
        })


def test_prediction_item_from_probability():
    """Verify classification logic, rounding, and label assignment."""
    # High probability -> Specialty
    item_spec = PredictionItem.from_probability(0.85236)
    assert item_spec.is_specialty == 1
    assert item_spec.probability == 0.8524
    assert item_spec.confidence == 0.8524
    assert item_spec.label == "Specialty"

    # Low probability -> Commercial
    item_comm = PredictionItem.from_probability(0.12344)
    assert item_comm.is_specialty == 0
    assert item_comm.probability == 0.1234
    assert item_comm.confidence == 0.8766
    assert item_comm.label == "Commercial"

    # Boundary clamping
    clamped_low = PredictionItem.from_probability(-0.5)
    assert clamped_low.probability == 0.0
    assert clamped_low.confidence == 1.0
    assert clamped_low.label == "Commercial"

    clamped_high = PredictionItem.from_probability(1.5)
    assert clamped_high.probability == 1.0
    assert clamped_high.confidence == 1.0
    assert clamped_high.label == "Specialty"


def test_prediction_response_model():
    """Verify PredictionResponse assembly and serialization."""
    item = PredictionItem.from_probability(0.9)
    res = PredictionResponse(
        predictions=[item],
        model_version="champion",
        count=1,
        inference_latency_ms=12.5,
        timestamp_utc="2026-11-16T05:00:00Z",
    )
    assert res.count == 1
    assert res.predictions[0].label == "Specialty"


def test_training_trigger_schemas():
    """Verify training request and response schemas."""
    req = TrainingTriggerRequest(model_type="lightgbm", register_as_champion=False)
    assert req.model_type == "lightgbm"
    assert req.register_as_champion is False

    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrainingTriggerRequest.model_validate({"extra_arg": "bad"})

    res = TrainingTriggerResponse(
        status="completed",
        run_id="run-123",
        model_type="random_forest",
        promoted_as_champion=True,
        metrics={"roc_auc": 0.791, "macro_f1": 0.664},
        timestamp_utc="2026-11-16T05:00:00Z",
    )
    assert res.run_id == "run-123"
    assert res.metrics["roc_auc"] == 0.791
