"""Unit tests for coffee data validation schemas and data contracts."""

from datetime import date

from pydantic import ValidationError
import pytest

from coffee_quality.data.schema import CoffeeFeatures, CoffeeRecord


def test_valid_coffee_record_from_csv_aliases() -> None:
    """Verify parsing a valid record using CQI CSV dot-separated aliases."""
    raw_payload = {
        "Country.of.Origin": "Ethiopia",
        "Variety": "Yirgacheffe",
        "Processing.Method": "Washed / Wet",
        "Color": "Green",
        "altitude_mean_meters": 2075.0,
        "Moisture": 0.12,
        "Category.One.Defects": 0,
        "Category.Two.Defects": 1,
        "Quakers": 0,
        "Grading.Date": "2015-04-04",
        "Is_Specialty": 1,
    }

    record = CoffeeRecord.model_validate(raw_payload)

    assert record.country_of_origin == "Ethiopia"
    assert record.variety == "Yirgacheffe"
    assert record.processing_method == "Washed / Wet"
    assert record.color == "Green"
    assert record.altitude_mean_meters == 2075.0
    assert record.moisture == 0.12
    assert record.category_one_defects == 0
    assert record.category_two_defects == 1
    assert record.quakers == 0
    assert record.grading_date == date(2015, 4, 4)
    assert record.is_specialty == 1


def test_valid_coffee_features_from_snake_case() -> None:
    """Verify instantiating features using native Python snake_case keyword arguments."""
    features = CoffeeFeatures(
        country_of_origin="Colombia",
        variety="Caturra",
        processing_method="Natural / Dry",
        color="Bluish-Green",
        altitude_mean_meters=1750.0,
        moisture=0.11,
        category_one_defects=1,
        category_two_defects=2,
        quakers=1,
        grading_date=date(2016, 5, 20),
    )

    assert features.country_of_origin == "Colombia"
    assert features.variety == "Caturra"
    assert features.altitude_mean_meters == 1750.0
    assert features.grading_date == date(2016, 5, 20)


def test_coffee_features_defaults() -> None:
    """Verify default values when instantiating features with minimal parameters."""
    features = CoffeeFeatures()

    assert features.country_of_origin == "Unknown"
    assert features.variety == "Unknown"
    assert features.processing_method == "Unknown"
    assert features.color == "Unknown"
    assert features.altitude_mean_meters == 1310.6
    assert features.moisture == 0.10
    assert features.category_one_defects == 0
    assert features.category_two_defects == 0
    assert features.quakers == 0
    assert features.grading_date == date.today()


def test_reject_negative_defects() -> None:
    """Verify that negative defect counts are rejected by schema boundaries."""
    with pytest.raises(ValidationError):
        CoffeeFeatures(category_one_defects=-1)

    with pytest.raises(ValidationError):
        CoffeeFeatures(category_two_defects=-1)

    with pytest.raises(ValidationError):
        CoffeeFeatures(quakers=-1)


def test_reject_out_of_bounds_altitude() -> None:
    """Verify elevation constraints rejecting negative values and values exceeding 4000m."""
    with pytest.raises(ValidationError):
        CoffeeFeatures(altitude_mean_meters=-10.0)

    with pytest.raises(ValidationError):
        CoffeeFeatures(altitude_mean_meters=4500.0)


def test_reject_out_of_bounds_moisture() -> None:
    """Verify moisture constraints rejecting negative percentages and ratios above 1.0."""
    with pytest.raises(ValidationError):
        CoffeeFeatures(moisture=-0.05)

    with pytest.raises(ValidationError):
        CoffeeFeatures(moisture=1.2)


def test_reject_invalid_date_format() -> None:
    """Verify that malformed date strings fail validation."""
    with pytest.raises(ValidationError):
        CoffeeFeatures.model_validate({"Grading.Date": "invalid-date"})

    with pytest.raises(ValidationError):
        CoffeeFeatures.model_validate({"Grading.Date": "04/04/2015"})


def test_reject_invalid_target_values() -> None:
    """Verify that is_specialty is strictly bounded to binary 0 or 1."""
    with pytest.raises(ValidationError):
        CoffeeRecord.model_validate({"Grading.Date": "2015-04-04", "Is_Specialty": 2})

    with pytest.raises(ValidationError):
        CoffeeRecord.model_validate({"Grading.Date": "2015-04-04", "Is_Specialty": -1})

    with pytest.raises(ValidationError):
        CoffeeRecord.model_validate({"Grading.Date": "2015-04-04", "Is_Specialty": "specialty"})


def test_reject_anti_leakage_extra_fields() -> None:
    """Verify that post-cupping sensory scores are strictly rejected by extra='forbid'."""
    payload_with_leakage = {
        "Country.of.Origin": "Guatemala",
        "Grading.Date": "2015-04-04",
        "Total.Cup.Points": 84.5,
    }

    with pytest.raises(ValidationError):
        CoffeeFeatures.model_validate(payload_with_leakage)


def test_reject_empty_string_features() -> None:
    """Verify min_length constraint rejecting empty string values."""
    with pytest.raises(ValidationError):
        CoffeeFeatures(country_of_origin="")
