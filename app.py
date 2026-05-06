import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("models/model.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Student(BaseModel):
    study_hours: int
    attendance: int
    previous_marks: int


@app.post("/predict")
def predict(data: Student):
    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": "PASS" if prediction == 1 else "FAIL",
        "confidence": f"{round(prob * 100, 2)}%"
    }