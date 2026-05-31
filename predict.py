import joblib
import pandas as pd

model = joblib.load("student_model.joblib")

sample = pd.DataFrame([{
    "attendance_pct": 65,
    "study_hours_wk": 5,
    "quiz_avg": 50,
    "assign_avg": 55,
    "midterm": 45,
    "lms_logins_wk": 2,
    "forum_posts": 1
}])

prediction = model.predict(sample)

print("Prediction:", prediction[0])