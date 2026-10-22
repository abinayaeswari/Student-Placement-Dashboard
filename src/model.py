"""
model.py
Trains and evaluates classification models to predict Placement_Status.
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)

df = pd.read_csv(str(DATA_DIR / "placement_data.csv"))
CHARTS = str(CHARTS_DIR)

# --- Prepare features ---
target = (df["Placement_Status"] == "Placed").astype(int)

cat_cols = ["Gender", "Branch", "Placement_Training", "Work_Experience"]
num_cols = ["SSC_Marks", "HSC_Marks", "CGPA", "IQ", "Internships", "Projects",
            "Certifications", "Aptitude_Test_Score", "Communication_Skills",
            "Extra_Curricular_Score", "Backlogs"]

X = df[cat_cols + num_cols].copy()
encoders = {}
for c in cat_cols:
    encoders[c] = LabelEncoder()
    X[c] = encoders[c].fit_transform(X[c])

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.2, random_state=42, stratify=target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# --- Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
pred_lr = log_reg.predict(X_test_scaled)
proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, pred_lr),
    "auc": roc_auc_score(y_test, proba_lr),
    "pred": pred_lr,
    "proba": proba_lr,
}

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, pred_rf),
    "auc": roc_auc_score(y_test, proba_rf),
    "pred": pred_rf,
    "proba": proba_rf,
}

print("=== Model Performance ===")
for name, r in results.items():
    print(f"\n{name}")
    print(f"  Accuracy: {r['accuracy']:.3f}")
    print(f"  ROC-AUC : {r['auc']:.3f}")
    print(classification_report(y_test, r["pred"], target_names=["Not Placed", "Placed"]))

# --- Confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"])
    ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{CHARTS}/09_confusion_matrices.png")
plt.close()

# --- ROC curves ---
plt.figure(figsize=(6, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS}/10_roc_curve.png")
plt.close()

# --- Feature importance (Random Forest) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#6A1B9A")
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/11_feature_importance.png")
plt.close()

print("\nTop 5 important features:\n", importances.head(5))

# Save model comparison summary
summary = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [r["accuracy"] for r in results.values()],
    "ROC_AUC": [r["auc"] for r in results.values()],
})
summary.to_csv(str(DATA_DIR / "model_results.csv"), index=False)
print("\nSaved model_results.csv")

# --- Save trained models so the Streamlit dashboard can reuse them ---
import joblib
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(
    {
        "logistic_regression": log_reg,
        "random_forest": rf,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": list(X.columns),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "metrics": {n: {"accuracy": float(r["accuracy"]), "roc_auc": float(r["auc"])} for n, r in results.items()},
    },
    MODELS_DIR / "placement_model.joblib",
)
print("Saved trained models to", MODELS_DIR / "placement_model.joblib")
