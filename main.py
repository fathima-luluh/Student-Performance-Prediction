from fastapi import FastAPI
import joblib

app = FastAPI(title="Student Performance Prediction API")

# Load trained model
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "API Running Successfully"}

@app.get("/predict")
def predict(
    study_hours: int,
    attendance: int,
    previous_marks: int,
    assignments_completed: int,
    sleep_hours: int
):
    # Feature engineering
    study_efficiency = study_hours / sleep_hours
    engagement_score = (attendance + assignments_completed * 10) / 2

    input_data = [[
        study_hours,
        attendance,
        previous_marks,
        assignments_completed,
        sleep_hours,
        study_efficiency,
        engagement_score
    ]]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 2)
    }