"""
Schema definitions for data validation using Pydantic.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np

class HousingData(BaseModel):
    """Schema for validating housing data inputs."""
    CRIM: float = Field(..., description="Per capita crime rate by town")
    ZN: float = Field(..., description="Proportion of residential land zoned for large lots")
    INDUS: float = Field(..., description="Proportion of non-retail business acres per town")
    CHAS: float = Field(..., description="Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    NOX: float = Field(..., description="Nitric oxides concentration (parts per 10 million)")
    RM: float = Field(..., description="Average number of rooms per dwelling")
    AGE: float = Field(..., description="Proportion of owner-occupied units built prior to 1940")
    DIS: float = Field(..., description="Weighted distances to employment centers")
    RAD: float = Field(..., description="Index of accessibility to radial highways")
    TAX: float = Field(..., description="Full-value property-tax rate per $10,000")
    PTRATIO: float = Field(..., description="Pupil-teacher ratio by town")
    B: float = Field(..., description="1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
    LSTAT: float = Field(..., description="% lower status of the population")
    Price: Optional[float] = Field(None, description="Median value of owner-occupied homes in $1000's")

    @validator('CRIM')
    def validate_crime_rate(cls, v):
        if v < 0:
            raise ValueError("Crime rate cannot be negative")
        return v

    @validator('ZN')
    def validate_zn(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Residential land proportion must be between 0 and 100")
        return v

    @validator('NOX')
    def validate_nox(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("NOX concentration must be between 0 and 1")
        return v

    @validator('RM')
    def validate_rooms(cls, v):
        if v <= 0:
            raise ValueError("Number of rooms must be positive")
        return v

    @validator('AGE', 'LSTAT')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Percentage values must be between 0 and 100")
        return v

    class Config:
        schema_extra = {
            "example": {
                "CRIM": 0.00632,
                "ZN": 18.0,
                "INDUS": 2.31,
                "CHAS": 0,
                "NOX": 0.538,
                "RM": 6.575,
                "AGE": 65.2,
                "DIS": 4.09,
                "RAD": 1,
                "TAX": 296,
                "PTRATIO": 15.3,
                "B": 396.9,
                "LSTAT": 4.98,
                "Price": 24.0
            }
        }

class HousingDataBatch(BaseModel):
    """Schema for validating batch housing data."""
    data: List[HousingData]

    @validator('data')
    def validate_batch_size(cls, v):
        if not v:
            raise ValueError("Batch cannot be empty")
        return v
