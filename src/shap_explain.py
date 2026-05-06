import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

print("🚀 Starting SHAP script...")

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

print("📦 Loading model...")
model = joblib.load("models/model.pkl")

print("📊 Loading data...")
df = pd.read_csv("data/students.csv")

X = df[[
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_completed",
    "sleep_hours",
    "study_efficiency",
    "engagement_score"
]]

print("🧠 Creating explainer...")
explainer = shap.TreeExplainer(model)

sample = X.iloc[:50]

print("⚡ Calculating SHAP values...")
shap_values = explainer.shap_values(sample)

print("📈 Saving summary plot...")
plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.close()

print("📊 Saving bar plot...")
plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.savefig("outputs/shap_bar.png", bbox_inches="tight")
plt.close()

print("✅ SHAP images saved successfully!")