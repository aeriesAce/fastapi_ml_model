from pydantic import BaseModel, Field

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=4, lt=8.5)
    sepal_width: float = Field(gt=1.8, lt=5)
    petal_length: float = Field(gt=0.8, lt=7.5)
    petal_width: float = Field(gt=0, lt=3)

class PredictionOutput(BaseModel):
    predicted_flower: str