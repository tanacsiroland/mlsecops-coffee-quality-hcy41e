"""Pydantic request and response schemas for the coffee quality prediction API."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CoffeePredictionInput(BaseModel):
    """Input payload for a single coffee sample prediction."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    country_of_origin: str = Field(
        default="Unknown",
        alias="Country.of.Origin",
        min_length=1,
        description="Country where the coffee lot was cultivated",
        examples=["Ethiopia"],
    )
    variety: str = Field(
        default="Unknown",
        alias="Variety",
        min_length=1,
        description="Botanical variety or cultivar of the coffee plant",
        examples=["Typica"],
    )
    processing_method: str = Field(
        default="Unknown",
        alias="Processing.Method",
        min_length=1,
        description="Post-harvest method used to process the coffee cherries",
        examples=["Washed / Wet"],
    )
    color: str = Field(
        default="Unknown",
        alias="Color",
        min_length=1,
        description="Visual color classification of the raw unroasted coffee beans",
        examples=["Green"],
    )
    altitude_mean_meters: float = Field(
        default=1310.6,
        alias="altitude_mean_meters",
        ge=0.0,
        le=4000.0,
        description="Mean site elevation in meters, bounded by Arabica cultivation limits",
        examples=[1800.0],
    )
    moisture: float = Field(
        default=0.10,
        alias="Moisture",
        ge=0.0,
        le=1.0,
        description="Moisture percentage of the green coffee beans (ratio between 0.0 and 1.0)",
        examples=[0.11],
    )
    category_one_defects: int = Field(
        default=0,
        alias="Category.One.Defects",
        ge=0,
        description="Count of primary full defects (e.g. black or sour beans)",
        examples=[0],
    )
    category_two_defects: int = Field(
        default=0,
        alias="Category.Two.Defects",
        ge=0,
        description="Count of secondary partial defects (e.g. insect damage, chipped beans)",
        examples=[2],
    )
    quakers: int = Field(
        default=0,
        alias="Quakers",
        ge=0,
        description="Count of unripe beans that fail to roast properly",
        examples=[0],
    )
    grading_date: str = Field(
        default="2020-01-01",
        alias="Grading.Date",
        min_length=1,
        description="Inspection date string (e.g. 'YYYY-MM-DD' or 'April 4th, 2015')",
        examples=["2020-01-01"],
    )


class CoffeeBatchRequest(BaseModel):
    """Batch prediction request payload."""

    model_config = ConfigDict(extra="forbid")

    items: list[CoffeePredictionInput] = Field(
        min_length=1,
        max_length=500,
        description="List of coffee samples to predict (1 to 500 items)",
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert batch items to a DataFrame with original feature column names."""
        records = [
            {
                "Country.of.Origin": item.country_of_origin,
                "Variety": item.variety,
                "Processing.Method": item.processing_method,
                "Color": item.color,
                "altitude_mean_meters": item.altitude_mean_meters,
                "Moisture": item.moisture,
                "Category.One.Defects": item.category_one_defects,
                "Category.Two.Defects": item.category_two_defects,
                "Quakers": item.quakers,
                "Grading.Date": str(item.grading_date),
            }
            for item in self.items
        ]
        return pd.DataFrame.from_records(records)


class PredictionItem(BaseModel):
    """Prediction result for an individual coffee sample."""

    is_specialty: int = Field(
        description="Binary prediction: 1 for Specialty Grade, 0 for Commercial Grade",
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated probability of the sample being Specialty Grade",
    )
    confidence: float = Field(
        ge=0.5,
        le=1.0,
        description="Confidence score for the predicted class: max(p, 1-p)",
    )
    label: str = Field(
        description="Human-readable classification label: 'Specialty' or 'Commercial'",
    )

    @classmethod
    def from_probability(cls, probability: float, threshold: float = 0.5) -> PredictionItem:
        """Construct a PredictionItem from positive class probability and decision threshold.

        Args:
            probability: Predicted probability of being Specialty grade.
            threshold: Decision boundary probability (default 0.5).

        Returns:
            Formatted PredictionItem instance.
        """
        prob_clamped = max(0.0, min(1.0, float(probability)))
        is_spec = 1 if prob_clamped >= threshold else 0
        confidence = prob_clamped if is_spec == 1 else (1.0 - prob_clamped)
        label = "Specialty" if is_spec == 1 else "Commercial"
        return cls(
            is_specialty=is_spec,
            probability=round(prob_clamped, 4),
            confidence=round(confidence, 4),
            label=label,
        )


class PredictionResponse(BaseModel):
    """Batch prediction response containing results and operational metadata."""

    predictions: list[PredictionItem] = Field(description="Individual prediction items")
    model_version: str = Field(description="Model registry version or identifier used for scoring")
    count: int = Field(description="Total count of predicted samples")
    inference_latency_ms: float = Field(description="Inference execution duration in milliseconds")
    timestamp_utc: str = Field(description="UTC timestamp of the prediction response")


class TrainingTriggerRequest(BaseModel):
    """Optional payload when triggering model retraining."""

    model_config = ConfigDict(extra="forbid")

    model_type: Literal["random_forest", "lightgbm"] = Field(
        default="random_forest",
        description="Model architecture to train ('random_forest' or 'lightgbm')",
    )
    register_as_champion: bool = Field(
        default=True,
        description="Whether to promote the winning model to 'champion' in the registry",
    )


class TrainingTriggerResponse(BaseModel):
    """Response payload after pipeline retraining completes."""

    status: str = Field(default="completed", description="Execution status")
    run_id: str = Field(description="MLflow run ID for the trained model")
    model_type: str = Field(description="Classifier family trained")
    promoted_as_champion: bool = Field(description="Whether the model was promoted to champion")
    metrics: dict[str, float] = Field(description="Holdout test evaluation metrics")
    timestamp_utc: str = Field(description="UTC timestamp of completion")
