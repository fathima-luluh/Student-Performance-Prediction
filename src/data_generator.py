import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "study_hours": np.random.randint(1, 10, n),
    "attendance": np.random.randint(50, 100, n),
    "previous_marks": np.random.randint(40, 100, n),
    "assignments_completed": np.random.randint(0, 10, n),
    "sleep_hours": np.random.randint(4, 10, n)
})

# Feature Engineering
data["study_efficiency"] = data["study_hours"] / data["sleep_hours"]
data["engagement_score"] = (data["attendance"] + data["assignments_completed"] * 10) / 2

# Final score
data["final_score"] = (
    0.25 * data["study_hours"] * 10 +
    0.25 * data["attendance"] +
    0.3 * data["previous_marks"] +
    0.2 * data["engagement_score"]
)

# Classification
data["pass"] = (data["final_score"] > 60).astype(int)

data.to_csv("data/students.csv", index=False)

print("✅ Advanced dataset created")