
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import predict

app = FastAPI(
    title="ML FastAPI Application",
    description="Iris classification API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four Iris measurements"
    )


@app.get("/")
def home():
    return {"message": "ML FastAPI application is running"}


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    try:
        prediction = predict(request.features)
        return {"prediction": prediction}
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
