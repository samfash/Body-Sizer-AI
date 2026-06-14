from pydantic import BaseModel
from typing import Dict

class HealthResponse(BaseModel):
    status: str

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
