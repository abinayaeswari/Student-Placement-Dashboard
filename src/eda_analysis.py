"""
eda_analysis.py
Exploratory Data Analysis on the Student Placement dataset.
Saves all charts as PNGs into ../charts/
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv(str(DATA_DIR / "placement_data.csv"))
CHARTS = str(CHARTS_DIR)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nPlacement rate:\n", df["Placement_Status"].value_counts(normalize=True))

# 1. Placement status count
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Placement_Status", palette=["#2E7D32", "#C62828"])
plt.title("Placement Status Distribution")
plt.tight_layout()
plt.savefig(f"{CHARTS}/01_placement_distribution.png")
plt.close()

# 2. CGPA distribution by placement
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Placement_Status", y="CGPA", palette=["#C62828", "#2E7D32"])
plt.title("CGPA vs Placement Status")
plt.tight_layout()
plt.savefig(f"{CHARTS}/02_cgpa_vs_placement.png")
plt.close()

# 3. Internships vs Placement
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df.groupby(["Internships", "Placement_Status"]).size().reset_index(name="count"),
    x="Internships", y="count", hue="Placement_Status"
)
plt.title("Internships Count vs Placement")
plt.tight_layout()
plt.savefig(f"{CHARTS}/03_internships_vs_placement.png")
plt.close()

# 4. Correlation heatmap (numeric features)
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(f"{CHARTS}/04_correlation_heatmap.png")
plt.close()

# 5. Branch-wise placement rate
branch_rate = (
    df.assign(Placed=(df["Placement_Status"] == "Placed").astype(int))
    .groupby("Branch")["Placed"].mean().sort_values(ascending=False) * 100
)
plt.figure(figsize=(6, 4))
branch_rate.plot(kind="bar", color="#1565C0")
plt.ylabel("Placement Rate (%)")
plt.title("Branch-wise Placement Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{CHARTS}/05_branch_placement_rate.png")
plt.close()

# 6. Communication skills vs Placement
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x="Placement_Status", y="Communication_Skills", palette=["#C62828", "#2E7D32"])
plt.title("Communication Skills vs Placement")
plt.tight_layout()
plt.savefig(f"{CHARTS}/06_communication_vs_placement.png")
plt.close()

# 7. Salary distribution (placed students)
plt.figure(figsize=(6, 4))
sns.histplot(df.loc[df["Placement_Status"] == "Placed", "Salary_LPA"], bins=20, kde=True, color="#2E7D32")
plt.title("Salary Distribution (Placed Students, LPA)")
plt.xlabel("Salary (LPA)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/07_salary_distribution.png")
plt.close()

# 8. Backlogs vs placement rate
backlog_rate = (
    df.assign(Placed=(df["Placement_Status"] == "Placed").astype(int))
    .groupby("Backlogs")["Placed"].mean() * 100
)
plt.figure(figsize=(6, 4))
backlog_rate.plot(kind="bar", color="#EF6C00")
plt.ylabel("Placement Rate (%)")
plt.title("Backlogs vs Placement Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{CHARTS}/08_backlogs_vs_placement.png")
plt.close()

print("\nAll charts saved to", CHARTS)

# Save a short text summary of key correlations with placement
df["Placed_Binary"] = (df["Placement_Status"] == "Placed").astype(int)
corr_with_target = numeric_df.assign(Placed_Binary=df["Placed_Binary"]).corr()["Placed_Binary"].sort_values(ascending=False)
corr_with_target.to_csv(str(DATA_DIR / "feature_correlation_with_placement.csv"))
print("\nTop correlated features with Placement:\n", corr_with_target)
