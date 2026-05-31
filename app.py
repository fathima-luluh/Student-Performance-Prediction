from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("student_model.joblib")

class Student(BaseModel):
    attendance_pct: int
    study_hours_wk: int
    quiz_avg: int
    assign_avg: int
    midterm: int
    lms_logins_wk: int
    forum_posts: int

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API"}

@app.post("/predict")
def predict(student: Student):

    data = pd.DataFrame([student.dict()])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "risk_probability": float(probability)
    }