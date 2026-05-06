import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/model.pkl")

# Feature names (IMPORTANT: same order as training)
features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_completed",
    "sleep_hours",
    "study_efficiency",
    "engagement_score"
]

# Get importance
importance = model.feature_importances_

# Create dataframe
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# Print
print(feat_df)

# Plot
plt.figure(figsize=(8,5))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()

plt.savefig("outputs/feature_importance.png")
plt.show()