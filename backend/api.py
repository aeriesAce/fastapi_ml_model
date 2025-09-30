from fastapi import FastAPI, APIRouter
from utils.constants import DATA_PATH, MODELS_PATH
from backend.data_processing import PredictionOutput, IrisInput
import pandas as pd
import joblib

df = pd.read_csv(DATA_PATH/ "IRIS.csv")
router = APIRouter(prefix = "/api/iris/v1")
app = FastAPI()


@router.get("")
def read_data():
    return df.to_dict(orient = "records")

@router.post("/predict", response_model = PredictionOutput)
def predict_flower(payload: IrisInput):
    data_to_predict = pd.DataFrame(payload.model_dump(), index = [0])
    clf = joblib.load(MODELS_PATH / "iris_classifier.joblib")
    prediction = clf.predict(data_to_predict)
    print(prediction)
    return {"predicted_flower": prediction[0]}

app.include_router(router=router)