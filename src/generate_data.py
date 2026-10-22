"""
generate_data.py
Generates a synthetic but realistic Student Placement dataset.
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"
import numpy as np
import pandas as pd

np.random.seed(42)
N = 800

# --- Academic & skill features ---
cgpa = np.clip(np.random.normal(7.2, 1.0, N), 4.5, 10.0).round(2)
iq = np.clip(np.random.normal(105, 15, N), 70, 150).round(0)
ssc_marks = np.clip(np.random.normal(78, 10, N), 40, 100).round(1)
hsc_marks = np.clip(np.random.normal(75, 10, N), 40, 100).round(1)
internships = np.random.poisson(1.2, N).clip(0, 5)
projects = np.random.poisson(2.0, N).clip(0, 8)
certifications = np.random.poisson(1.5, N).clip(0, 6)
aptitude_score = np.clip(np.random.normal(65, 15, N), 20, 100).round(1)
communication_skills = np.clip(np.random.normal(6.5, 1.5, N), 1, 10).round(1)
extra_curricular = np.clip(np.random.normal(5.5, 2.0, N), 0, 10).round(1)
backlogs = np.random.choice([0, 1, 2, 3], N, p=[0.65, 0.2, 0.1, 0.05])
placement_training = np.random.choice(["Yes", "No"], N, p=[0.6, 0.4])
work_experience = np.random.choice(["Yes", "No"], N, p=[0.25, 0.75])
branch = np.random.choice(
    ["CSE", "ECE", "MECH", "CIVIL", "EEE", "IT"], N,
    p=[0.30, 0.20, 0.15, 0.10, 0.10, 0.15]
)
gender = np.random.choice(["Male", "Female"], N, p=[0.58, 0.42])

# --- Placement probability as a weighted logistic function ---
z = (
    -0.5  # intercept, keeps overall placement rate realistic (~55-60%)
    + 0.55 * (cgpa - 7)
    + 0.02 * (iq - 105)
    + 0.35 * internships
    + 0.20 * projects
    + 0.30 * certifications
    + 0.015 * (aptitude_score - 65)
    + 0.30 * (communication_skills - 6.5)
    + 0.10 * (extra_curricular - 5.5)
    - 0.45 * backlogs
    + 0.35 * (placement_training == "Yes")
    + 0.25 * (work_experience == "Yes")
    + np.random.normal(0, 0.8, N)  # noise
)
prob = 1 / (1 + np.exp(-z))
placement = (prob > 0.5).astype(int)

# --- Salary only for placed students ---
base_salary = (
    3.0
    + 0.55 * (cgpa - 6)
    + 0.25 * internships
    + 0.20 * projects
    + 0.15 * certifications
    + 0.02 * (aptitude_score - 60)
    + (branch == "CSE") * 1.2
    + (branch == "IT") * 0.9
    + (branch == "ECE") * 0.4
    + np.random.normal(0, 0.6, N)
)
salary = np.where(placement == 1, np.clip(base_salary, 2.5, 20).round(2), np.nan)

df = pd.DataFrame({
    "Student_ID": [f"S{i+1:04d}" for i in range(N)],
    "Gender": gender,
    "Branch": branch,
    "SSC_Marks": ssc_marks,
    "HSC_Marks": hsc_marks,
    "CGPA": cgpa,
    "IQ": iq,
    "Internships": internships,
    "Projects": projects,
    "Certifications": certifications,
    "Aptitude_Test_Score": aptitude_score,
    "Communication_Skills": communication_skills,
    "Extra_Curricular_Score": extra_curricular,
    "Backlogs": backlogs,
    "Placement_Training": placement_training,
    "Work_Experience": work_experience,
    "Placement_Status": np.where(placement == 1, "Placed", "Not Placed"),
    "Salary_LPA": salary,
})

df.to_csv(str(DATA_DIR / "placement_data.csv"), index=False)
print("Dataset generated:", df.shape)
print(df["Placement_Status"].value_counts())
