import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "attendance_pct": np.random.randint(40, 100, n),
    "study_hours_wk": np.random.randint(1, 30, n),
    "quiz_avg": np.random.randint(30, 100, n),
    "assign_avg": np.random.randint(30, 100, n),
    "midterm": np.random.randint(30, 100, n),
    "lms_logins_wk": np.random.randint(0, 20, n),
    "forum_posts": np.random.randint(0, 15, n),
}

df = pd.DataFrame(data)

score = (
    df["attendance_pct"] * 0.25 +
    df["study_hours_wk"] * 1.5 +
    df["quiz_avg"] * 0.2 +
    df["assign_avg"] * 0.2 +
    df["midterm"] * 0.3
)

df["passed"] = (score > 60).astype(int)

df.to_csv("students.csv", index=False)

print("Dataset Created Successfully")
print(df.head())