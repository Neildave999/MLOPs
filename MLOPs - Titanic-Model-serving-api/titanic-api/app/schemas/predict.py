from typing import Any, List, Optional

from pydantic import BaseModel
from regression_model.processing.validation import TitanicDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "pclass": 1,
                        "sex": "female",
                        "age": 29,
                        "sibsp": 0,
                        "parch": 0,
                        "fare": 211.3375, 
                        "cabin": "B5",
                        "embarked": "S",
                        "title": "Miss",
                    }
                ]
            }
        }
