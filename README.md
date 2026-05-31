\# Student Performance Prediction System



\## Overview



This project predicts student academic performance using Machine Learning techniques.



The system analyzes student-related factors such as:



\- Attendance

\- Study Hours

\- Quiz Scores

\- Assignment Scores

\- Midterm Marks

\- LMS Activity



and predicts whether a student is likely to pass or fail.



\---



\# Features



\- Synthetic student dataset generation

\- Machine Learning prediction model

\- FastAPI backend API

\- Student risk prediction

\- Model evaluation metrics

\- Data visualizations

\- Feature importance analysis



\---



\# Tech Stack



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- FastAPI

\- Matplotlib



\---



\# Machine Learning Workflow



Student Data

→ Data Preprocessing

→ Model Training

→ Prediction

→ Risk Analysis



\---



\# Model Used



\- Random Forest Classifier



\---



\# Accuracy



Model Accuracy: 94%



\---



\# API Endpoint



POST `/predict`



Example JSON:



```json

{

&#x20; "attendance\_pct": 85,

&#x20; "study\_hours\_wk": 15,

&#x20; "quiz\_avg": 78,

&#x20; "assign\_avg": 80,

&#x20; "midterm": 74,

&#x20; "lms\_logins\_wk": 10,

&#x20; "forum\_posts": 5

}

