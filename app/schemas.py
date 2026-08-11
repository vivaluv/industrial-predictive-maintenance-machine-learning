from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class PredictionRequest(BaseModel):
    """
    Input schema for machine failure prediction.

    Machine type uses one-hot encoding with Low
    as the reference category:

    Low:
        Type_M = 0
        Type_H = 0

    Medium:
        Type_M = 1
        Type_H = 0

    High:
        Type_M = 0
        Type_H = 1
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Air_temperature_K": 300.2,
                "Process_temperature_K": 309.4,
                "Rotational_speed_rpm": 1500,
                "Torque_Nm": 42.5,
                "Tool_wear_min": 120,
                "Type_M": 1,
                "Type_H": 0,
            }
        },
    )

    Air_temperature_K: float = Field(
        ...,
        description="Air temperature in Kelvin.",
        examples=[300.2],
    )

    Process_temperature_K: float = Field(
        ...,
        description="Process temperature in Kelvin.",
        examples=[309.4],
    )

    Rotational_speed_rpm: int = Field(
        ...,
        ge=0,
        description=(
            "Machine rotational speed in "
            "revolutions per minute (RPM)."
        ),
        examples=[1500],
    )

    Torque_Nm: float = Field(
        ...,
        ge=0,
        description=(
            "Machine torque in Newton metres (Nm)."
        ),
        examples=[42.5],
    )

    Tool_wear_min: int = Field(
        ...,
        ge=0,
        description=(
            "Accumulated tool wear in minutes."
        ),
        examples=[120],
    )

    Type_M: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Medium-quality machine indicator "
            "(0 or 1)."
        ),
        examples=[1],
    )

    Type_H: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "High-quality machine indicator "
            "(0 or 1)."
        ),
        examples=[0],
    )

    @model_validator(mode="after")
    def validate_machine_type(self):
        """
        Ensure machine-type dummy variables form
        a valid one-hot encoding.

        Low is represented by both indicators being 0.
        """

        if self.Type_M == 1 and self.Type_H == 1:
            raise ValueError(
                "Type_M and Type_H cannot both "
                "equal 1."
            )

        return self


class TopContributor(BaseModel):
    """
    One local SHAP feature contribution.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    feature: str = Field(
        ...,
        description="Feature name.",
    )

    feature_value: float = Field(
        ...,
        description=(
            "Observed value of the feature."
        ),
    )

    shap_value: float = Field(
        ...,
        description=(
            "Signed SHAP contribution for "
            "the feature."
        ),
    )

    absolute_shap: float = Field(
        ...,
        ge=0,
        description=(
            "Absolute magnitude of the SHAP "
            "contribution."
        ),
    )


class PredictionResponse(BaseModel):
    """
    Output schema returned by the prediction API.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    predicted_class: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Model classification "
            "(0 = No Failure, 1 = Failure)."
        ),
    )

    failure_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Predicted probability of "
            "machine failure."
        ),
    )

    decision_threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Operational probability threshold "
            "used for the maintenance decision."
        ),
    )

    decision_prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Threshold-based operational decision "
            "(0 = Normal Operation, "
            "1 = Maintenance Required)."
        ),
    )

    top_contributors: list[
        TopContributor
    ] = Field(
        ...,
        description=(
            "Most influential local SHAP "
            "contributors for the prediction."
        ),
    )