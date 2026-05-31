import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("students.csv")

# Features and target
X = df.drop("passed", axis=1)
y = df["passed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, preds)

print("Accuracy:", accuracy)

# -------------------------------
# 1. Correlation Heatmap
# -------------------------------

plt.figure(figsize=(8,6))

corr = df.corr()

plt.imshow(corr, cmap='coolwarm')

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.close()

# -------------------------------
# 2. Confusion Matrix
# -------------------------------

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.close()

# -------------------------------
# 3. ROC Curve
# -------------------------------

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("ROC Curve")

plt.savefig("roc_curve.png")

plt.close()

# -------------------------------
# 4. Feature Importance
# -------------------------------

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()

print("All visualizations saved successfully.")