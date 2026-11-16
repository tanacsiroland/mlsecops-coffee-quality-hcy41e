"""Data schemas and contract models for coffee lot attributes.

Enforces strict input validation, physical bounds checking, and anti-leakage
controls on both training datasets and live inference requests.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CoffeeFeatures(BaseModel):
    """Predictive features available prior to sensory cupping."""

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
    )
    variety: str = Field(
        default="Unknown",
        alias="Variety",
        min_length=1,
        description="Botanical variety or cultivar of the coffee plant",
    )
    processing_method: str = Field(
        default="Unknown",
        alias="Processing.Method",
        min_length=1,
        description="Post-harvest method used to process the coffee cherries",
    )
    color: str = Field(
        default="Unknown",
        alias="Color",
        min_length=1,
        description="Visual color classification of the raw unroasted coffee beans",
    )
    altitude_mean_meters: float = Field(
        default=1310.6,
        alias="altitude_mean_meters",
        ge=0.0,
        le=4000.0,
        description="Mean site elevation in meters, bounded by Arabica cultivation limits",
    )
    moisture: float = Field(
        default=0.10,
        alias="Moisture",
        ge=0.0,
        le=1.0,
        description="Moisture percentage of the green coffee beans (ratio between 0.0 and 1.0)",
    )
    category_one_defects: int = Field(
        default=0,
        alias="Category.One.Defects",
        ge=0,
        description="Count of primary full defects (e.g. black or sour beans)",
    )
    category_two_defects: int = Field(
        default=0,
        alias="Category.Two.Defects",
        ge=0,
        description="Count of secondary partial defects (e.g. insect damage, chipped beans)",
    )
    quakers: int = Field(
        default=0,
        alias="Quakers",
        ge=0,
        description="Count of unripe beans that fail to roast properly",
    )
    grading_date: date = Field(
        default_factory=date.today,
        alias="Grading.Date",
        description="Inspection and grading date in ISO 8601 format (YYYY-MM-DD)",
    )


class CoffeeRecord(CoffeeFeatures):
    """Full coffee sample record including the supervised classification target."""

    is_specialty: Literal[0, 1] = Field(
        alias="Is_Specialty",
        description="Binary classification target: 1 for Specialty Grade (>=80.0), 0 for Commercial Grade (<80.0)",
    )
