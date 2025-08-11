"""
Schema definitions for data validation using Pydantic.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np

class HousingData(BaseModel):
    """Schema for validating housing data inputs."""
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")
    Price: Optional[float] = Field(None, description="Median house value in block group")

    @validator('MedInc')
    def validate_income(cls, v):
        if v < 0:
            raise ValueError("Median income cannot be negative")
        return v

    @validator('HouseAge')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError("House age cannot be negative")
        return v

    @validator('AveRooms', 'AveBedrms')
    def validate_rooms(cls, v):
        if v <= 0:
            raise ValueError("Number of rooms must be positive")
        return v

    @validator('Population')
    def validate_population(cls, v):
        if v < 0:
            raise ValueError("Population cannot be negative")
        return v

    @validator('AveOccup')
    def validate_occupancy(cls, v):
        if v <= 0:
            raise ValueError("Average occupancy must be positive")
        return v

    @validator('Latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @validator('Longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23,
                "Price": 4.526
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
