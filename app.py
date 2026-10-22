"""
Student Placement Analysis Dashboard
Run with:  python -m streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import io
from reportlab.pdfgen import canvas

from src.predict import (
    MODEL_LABELS,
    ModelNotFoundError,
    feature_importance,
    load_bundle,
    predict_student,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "placement_data.csv"
RESULTS_PATH = BASE_DIR / "data" / "model_results.csv"
CORR_PATH = BASE_DIR / "data" / "feature_correlation_with_placement.csv"
CHARTS_DIR = BASE_DIR / "charts"
REPORT_PATH = BASE_DIR / "assets" / "Student_Placement_Data_Analysis_Report.docx"

st.set_page_config(
    page_title="Student Placement Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------
# Login and Registration System
# ----------------------------------------------------------------------------

import json

USERS_FILE = BASE_DIR / "data" / "users.json"

if not USERS_FILE.exists():
    USERS_FILE.write_text(
        json.dumps({}, indent=4),
        encoding="utf-8"
    )

users = json.loads(USERS_FILE.read_text(encoding="utf-8"))

if "S0001" not in users:
    users["S0001"] = {
        "name": "Student",
        "password": "student123",
        "role": "Student",
        "department": "",
        "email": "",
        "phone": "",
        "cgpa": 0.0
    }

    USERS_FILE.write_text(
        json.dumps(users, indent=4),
        encoding="utf-8"
    )

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

if not st.session_state.logged_in:
    st.title("🎓 Student Placement Portal")

    tab1, tab2 = st.tabs(["Login", "Student Register"])

    with tab1:
        role = st.selectbox("Select Login Type", ["Student", "Admin"])

        with st.form("login_form"):
            username = st.text_input("Username / Register Number")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")

        if login:
            if role == "Admin" and username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.role = "Admin"
                st.rerun()

            elif role == "Student" and username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role = "Student"
                st.session_state.username = username
                st.rerun()

            else:
                st.error("Invalid Username or Password")

    with tab2:
        st.subheader("Create Student Account")

        with st.form("register_form"):
            name = st.text_input("Student Name")
            reg_no = st.text_input("Register Number")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register = st.form_submit_button("Register")

        if register:
            if not name.strip() or not reg_no.strip() or not new_password:
                st.error("Please fill all fields")
            elif reg_no in users:
                st.error("Register Number already exists")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                users[reg_no] = {
                    "name": name.strip(),
                    "password": new_password,
                    "role": "Student"
                }
                USERS_FILE.write_text(
                    json.dumps(users, indent=4),
                    encoding="utf-8"
                )
                st.success("Registration successful! Please login.")

    st.stop()
# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
:root {
    --bg: #0b1220;
    --panel: #111b30;
    --panel-2: #16233f;
    --border: #22345c;
    --blue: #3b82f6;
    --blue-soft: #60a5fa;
    --purple: #8b5cf6;
    --text: #e6edf9;
    --muted: #93a4c3;
    --green: #22c55e;
    --red: #f43f5e;
}
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,.14), transparent 60%), var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.01em; }
p, li, label, span { color: var(--text); }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1730 0%, #0a1226 100%); border-right: 1px solid var(--border); }
.sb-brand { display:flex; align-items:center; gap:.6rem; padding:.4rem .2rem 1rem; }
.sb-logo { width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center;
           background: linear-gradient(135deg, var(--blue), var(--purple)); font-size:1.3rem; }
.sb-title { font-weight:700; font-size:1.02rem; line-height:1.15; color:var(--text); }
.sb-sub { font-size:.74rem; color:var(--muted); }

/* Hero */
.hero { background: linear-gradient(135deg, rgba(59,130,246,.22), rgba(139,92,246,.20));
        border:1px solid var(--border); border-radius:20px; padding:2rem 2.2rem; margin-bottom:1.6rem; }
.hero h1 { margin:0 0 .4rem 0; font-size:2.05rem; }
.hero p { margin:0; color:var(--muted); font-size:1.02rem; max-width:820px; line-height:1.55; }
.badge { display:inline-block; font-size:.74rem; padding:.2rem .65rem; border-radius:999px; margin-right:.4rem; margin-top:.9rem;
         background: rgba(59,130,246,.16); border:1px solid rgba(96,165,250,.4); color:var(--blue-soft); }

/* Metric cards */
.cards { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:1rem; margin-bottom:1.6rem; }
@media (max-width: 900px) { .cards { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 520px) { .cards { grid-template-columns: 1fr; } }
.card { background: linear-gradient(160deg, var(--panel-2), var(--panel)); border:1px solid var(--border);
        border-radius:16px; padding:1.1rem 1.25rem; position:relative; overflow:hidden; }
.card::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--accent, var(--blue)); }
.card .label { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
.card .value { font-size:2.15rem; font-weight:700; margin-top:.25rem; color:var(--text); }
.card .note  { color:var(--muted); font-size:.78rem; margin-top:.1rem; }

/* Section panels */
.panel { background: var(--panel); border:1px solid var(--border); border-radius:16px; padding:1.1rem 1.3rem; margin-bottom:1rem; }
.section-title { font-size:1.15rem; font-weight:650; margin:.2rem 0 .8rem; color:var(--text); }
.caption { color:var(--muted); font-size:.82rem; margin:.3rem 0 1rem; text-align:center; }

/* Prediction result */
.result { border-radius:18px; padding:1.6rem 1.8rem; text-align:center; border:1px solid var(--border); }
.result.ok  { background: linear-gradient(135deg, rgba(34,197,94,.20), rgba(59,130,246,.12)); border-color: rgba(34,197,94,.55); }
.result.bad { background: linear-gradient(135deg, rgba(244,63,94,.20), rgba(139,92,246,.12)); border-color: rgba(244,63,94,.55); }
.result .verdict { font-size:2rem; font-weight:800; }
.result .sub { color:var(--muted); margin-top:.3rem; }

/* Widgets */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--blue), var(--purple)); color:#fff; border:none; border-radius:10px;
    font-weight:600; padding:.55rem 1.3rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover { filter:brightness(1.12); color:#fff; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
[data-testid="stImage"] img { border-radius:12px; border:1px solid var(--border); background:#fff; }
[data-testid="stMetric"] { background: var(--panel); border:1px solid var(--border); border-radius:14px; padding:.8rem 1rem; }
hr { border-color: var(--border); }
footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show_image(path: Path):
    """Display an image across Streamlit versions (width API changed over time)."""
    try:
        st.image(str(path), width="stretch")
    except TypeError:
        st.image(str(path), use_container_width=True)


def show_df(df: pd.DataFrame, height=None):
    kwargs = {"height": height} if height else {}
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def metric_cards(items):
    html = '<div class="cards">'
    for label, value, note, accent in items:
        html += (
            f'<div class="card" style="--accent:{accent}">'
            f'<div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="note">{note}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle()


def safe_load(path: Path, label: str):
    if not path.exists():
        st.error(f"{label} not found at `{path.relative_to(BASE_DIR)}`. Make sure the project folder is complete.")
        return None
    try:
        return load_csv(str(path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {label}: {exc}")
        return None


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
PAGES = [
    "🏠 Home Dashboard",
    "👤 Student Profile",
    "📊 Data Analysis",
    "🗂️ Student Data",
    "📅 Attendance Management",
    "🤖 ML Prediction"
]
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-logo">🎓</div><div>'
        '<div class="sb-title">Placement Analytics</div>'
        '<div class="sb-sub">Student Placement Data Analysis</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
    if REPORT_PATH.exists():
        try:
            st.download_button(
                "⬇️ Download project report",
                data=REPORT_PATH.read_bytes(),
                file_name=REPORT_PATH.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception:  # noqa: BLE001
            pass

df = safe_load(DATA_PATH, "Dataset")
if df is None:
    st.stop()

placed_mask = df["Placement_Status"] == "Placed"
total = len(df)
placed = int(placed_mask.sum())
not_placed = total - placed
placement_pct = placed / total * 100 if total else 0.0

# ----------------------------------------------------------------------------
# Student Profile
# ----------------------------------------------------------------------------


def page_profile():
    st.markdown("## 👤 Student Profile")
    st.caption("View and update student details.")

    is_admin = st.session_state.get("role") == "Admin"

    if is_admin:
        student_ids = [
            sid for sid, details in users.items()
            if details.get("role") == "Student"
        ]

        if not student_ids:
            st.info("No registered students found.")
            return

        selected_id = st.selectbox(
            "Select Student Register Number",
            student_ids,
            key="profile_student_select"
        )
        username = selected_id
    else:
        username = st.session_state.get("username", "")

    student = users.get(username, {})

    st.markdown("### Personal Information")

    with st.form("profile_form"):
        name = st.text_input(
            "Student Name",
            value=student.get("name", "")
        )

        st.text_input(
            "Register Number",
            value=username,
            disabled=True
        )

        department = st.text_input(
            "Department",
            value=student.get("department", "")
        )

        email = st.text_input(
            "Email ID",
            value=student.get("email", "")
        )

        phone = st.text_input(
            "Phone Number",
            value=student.get("phone", "")
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=float(student.get("cgpa", 0.0)),
            step=0.1
        )
        st.text_input("Role", value="Student", disabled=True)

        save = st.form_submit_button(
            "Update Student" if is_admin else "Save Profile"
        )

    if save:
        users[username].update({
            "name": name.strip(),
            "department": department.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "cgpa": cgpa
        })

        USERS_FILE.write_text(
            json.dumps(users, indent=4),
            encoding="utf-8"
        )

        st.success("Student details updated successfully!")
        st.rerun()
    # PDF Student Report
    student = users.get(username, {})

    st.markdown("### Student Report")

    report_buffer = io.BytesIO()
    pdf = canvas.Canvas(report_buffer)

    pdf.setTitle(f"Student Report - {username}")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 800, "Student Profile Report")

    pdf.setFont("Helvetica", 11)

    report_lines = [
        f"Register Number: {username}",
        f"Student Name: {student.get('name', '')}",
        f"Department: {student.get('department', '')}",
        f"Email: {student.get('email', '')}",
        f"Phone: {student.get('phone', '')}",
        f"CGPA: {student.get('cgpa', 0.0)}",
    ]

    y = 760
    for line in report_lines:
        pdf.drawString(50, y, line)
        y -= 25

    pdf.save()

    st.download_button(
        "Download Student Report (PDF)",
        data=report_buffer.getvalue(),
        file_name=f"{username}_student_report.pdf",
        mime="application/pdf"
    )

# ----------------------------------------------------------------------------
# Page 1: Home
# ----------------------------------------------------------------------------
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1>🎓 Student Placement Analysis Dashboard</h1>
          <p>An end-to-end data analysis and machine learning project that explores which academic,
          skill and experience factors influence campus placement outcomes, and predicts whether a
          student is likely to be placed.</p>
          <span class="badge">Python</span><span class="badge">Pandas</span>
          <span class="badge">Scikit-learn</span><span class="badge">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cards([
        ("Total Students", f"{total:,}", "Records in dataset", "#3b82f6"),
        ("Placed", f"{placed:,}", "Students placed", "#22c55e"),
        ("Not Placed", f"{not_placed:,}", "Students not placed", "#f43f5e"),
        ("Placement Rate", f"{placement_pct:.1f}%", "Placed ÷ Total", "#8b5cf6"),
    ])

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Placement rate by branch</div>', unsafe_allow_html=True)
        by_branch = (
            df.assign(Placed=placed_mask.astype(int))
            .groupby("Branch")["Placed"].mean().mul(100).round(1)
            .sort_values(ascending=False).rename("Placement rate (%)")
        )
        st.bar_chart(by_branch, color="#3b82f6")

    with right:
        st.markdown('<div class="section-title">Top 5 factors (Random Forest importance)</div>', unsafe_allow_html=True)
        try:
            bundle = get_bundle()
            top5 = feature_importance(bundle).head(5)
            maxv = float(top5.max())
            html = '<div class="panel">'
            for i, (name, val) in enumerate(top5.items(), 1):
                width = val / maxv * 100
                html += (
                    f'<div style="margin:.55rem 0"><div style="display:flex;justify-content:space-between;font-size:.9rem">'
                    f'<span>{i}. {name}</span><span style="color:#93a4c3">{val:.4f}</span></div>'
                    f'<div style="height:8px;background:#0b1220;border-radius:6px;margin-top:.25rem">'
                    f'<div style="height:8px;width:{width:.0f}%;border-radius:6px;'
                    f'background:linear-gradient(90deg,#3b82f6,#8b5cf6)"></div></div></div>'
                )
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        except ModelNotFoundError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load model: {exc}")

    st.markdown('<div class="section-title">Model performance (held-out test set)</div>', unsafe_allow_html=True)
    results = safe_load(RESULTS_PATH, "Model results")
    if results is not None:
        shown = results.copy()
        for col in ("Accuracy", "ROC_AUC"):
            if col in shown.columns:
                shown[col] = shown[col].round(3)
        show_df(shown)


# ----------------------------------------------------------------------------
# Page 2: Data Analysis (existing chart files only)
# ----------------------------------------------------------------------------
CHARTS = [
    ("01_placement_distribution.png", "Placement Distribution", "How many students were placed vs not placed."),
    ("02_cgpa_vs_placement.png", "CGPA vs Placement", "CGPA spread for placed and not placed students."),
    ("03_internships_vs_placement.png", "Internships vs Placement", "Placement outcome by number of internships."),
    ("04_correlation_heatmap.png", "Correlation Heatmap", "Correlation between all numeric features."),
    ("05_branch_placement_rate.png", "Branch Placement Rate", "Placement percentage for each branch."),
    ("06_communication_vs_placement.png", "Communication vs Placement", "Communication skill ratings by placement status."),
    ("07_salary_distribution.png", "Salary Distribution", "Salary (LPA) of placed students."),
    ("08_backlogs_vs_placement.png", "Backlogs vs Placement", "Placement rate by number of backlogs."),
    ("09_confusion_matrices.png", "Confusion Matrices", "Model predictions vs actual outcomes on the test set."),
    ("10_roc_curve.png", "ROC Curve", "ROC comparison of Logistic Regression and Random Forest."),
    ("11_feature_importance.png", "Feature Importance", "Random Forest feature importance ranking."),
]


def page_analysis():
    st.markdown("## 📊 Data Analysis")
    st.caption("All charts below are the generated image files from the `charts/` folder.")

    tab_eda, tab_model = st.tabs(["Exploratory Analysis (8 charts)", "Model Evaluation (3 charts)"])

    def render(subset):
        missing = [fn for fn, _, _ in subset if not (CHARTS_DIR / fn).exists()]
        if missing:
            st.warning("Missing chart files: " + ", ".join(missing))
        cols = st.columns(2, gap="large")
        for idx, (fn, title, desc) in enumerate(subset):
            path = CHARTS_DIR / fn
            with cols[idx % 2]:
                st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                if path.exists():
                    show_image(path)
                    st.markdown(f'<div class="caption">{desc}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"{fn} not found")

    with tab_eda:
        render(CHARTS[:8])
    with tab_model:
        render(CHARTS[8:])
        corr = safe_load(CORR_PATH, "Correlation file")
        if corr is not None:
            st.markdown('<div class="section-title">Correlation with placement (from CSV)</div>', unsafe_allow_html=True)
            show_df(corr)


# ----------------------------------------------------------------------------
# Page 3: Student data
# ----------------------------------------------------------------------------
def page_students():
    st.markdown("## 🗂️ Student Data")

    f1, f2, f3 = st.columns([2, 2, 2])
    branches = sorted(df["Branch"].dropna().unique().tolist())
    statuses = sorted(df["Placement_Status"].dropna().unique().tolist())
    with f1:
        sel_branch = st.multiselect("Department / Branch", branches, default=branches)
    with f2:
        sel_status = st.multiselect("Placement status", statuses, default=statuses)
    with f3:
        query = st.text_input("Search (Student ID, branch, gender…)", placeholder="e.g. S0012 or CSE")

    cg_min, cg_max = float(df["CGPA"].min()), float(df["CGPA"].max())
    cgpa_range = st.slider("CGPA range", cg_min, cg_max, (cg_min, cg_max), step=0.1)

    view = df[
        df["Branch"].isin(sel_branch)
        & df["Placement_Status"].isin(sel_status)
        & df["CGPA"].between(*cgpa_range)
    ]
    if query.strip():
        q = query.strip().lower()
        text_cols = view[["Student_ID", "Gender", "Branch", "Placement_Training", "Work_Experience", "Placement_Status"]]
        mask = text_cols.apply(lambda col: col.astype(str).str.lower().str.contains(q, regex=False)).any(axis=1)
        view = view[mask]

    n = len(view)
    n_placed = int((view["Placement_Status"] == "Placed").sum())
    avg_sal = view["Salary_LPA"].mean()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students shown", f"{n:,}")
    m2.metric("Placed", f"{n_placed:,}")
    m3.metric("Placement rate", f"{(n_placed / n * 100):.1f}%" if n else "–")
    m4.metric("Avg salary (LPA)", f"{avg_sal:.2f}" if pd.notna(avg_sal) else "–")

    if view.empty:
        st.info("No students match the current filters.")
        return

    show_df(view.reset_index(drop=True), height=420)
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_students.csv",
        mime="text/csv",
    )
    st.markdown('<div class="section-title">Summary statistics (filtered data)</div>', unsafe_allow_html=True)
    show_df(view.describe().T.round(2))

    # Admin Student Management
    st.markdown("---")
    st.markdown("## 🛠️ Admin Student Management")

    if st.session_state.get("role") == "Admin":

        student_ids = [
            sid for sid, details in users.items()
            if details.get("role") == "Student"
        ]

        if not student_ids:
            st.info("No registered students found.")
        else:
            selected_id = st.selectbox(
                "Select Student",
                student_ids
            )

            student = users[selected_id]

            with st.form("admin_edit_student"):
                name = st.text_input(
                    "Student Name",
                    value=student.get("name", "")
                )
                department = st.text_input(
                    "Department",
                    value=student.get("department", "")
                )
                email = st.text_input(
                    "Email",
                    value=student.get("email", "")
                )
                phone = st.text_input(
                    "Phone",
                    value=student.get("phone", "")
                )

                save = st.form_submit_button("Update Student")

            if save:
                users[selected_id].update({
                    "name": name.strip(),
                    "department": department.strip(),
                    "email": email.strip(),
                    "phone": phone.strip()
                })

                USERS_FILE.write_text(
                    json.dumps(users, indent=4),
                    encoding="utf-8"
                )

                st.success("Student details updated!")
                st.rerun()

            if st.button("Delete Selected Student", key="delete_student"):
                st.session_state["confirm_delete_student"] = selected_id

            if st.session_state.get("confirm_delete_student") == selected_id:
                st.warning(f"Are you sure you want to delete {selected_id}?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes, Delete", key="confirm_delete_yes"):
                        users.pop(selected_id, None)

                        USERS_FILE.write_text(
                            json.dumps(users, indent=4),
                            encoding="utf-8"
                        )

                        st.session_state.pop("confirm_delete_student", None)
                        st.success("Student deleted successfully!")
                        st.rerun()

                with col2:
                    if st.button("Cancel", key="confirm_delete_no"):
                        st.session_state.pop("confirm_delete_student", None)
                        st.rerun()

    else:
        st.info("Admin access only.")

# ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------
# Attendance Management
# ----------------------------------------------------------------------------

def page_attendance():
    st.markdown("## 📅 Attendance Management")
    st.caption("Admin can mark attendance. Students can view their attendance.")

    attendance_file = BASE_DIR / "data" / "attendance.json"

    if not attendance_file.exists():
        attendance_file.write_text("{}", encoding="utf-8")

    attendance = json.loads(
        attendance_file.read_text(encoding="utf-8")
    )

    is_admin = st.session_state.get("role") == "Admin"

    st.write("Current Role:", st.session_state.get("role"))
    st.write("Current Username:", st.session_state.get("username"))
    st.write("Admin Status:", is_admin)

    # Admin Attendance Entry
    if is_admin:
        student_ids = [
            sid for sid, details in users.items()
            if details.get("role") == "Student"
        ]

        if not student_ids:
            st.info("No registered students found.")
            return

        selected_id = st.selectbox(
            "Select Student",
            student_ids,
            key="attendance_student"
        )

        attendance_date = st.date_input("Attendance Date")

        status = st.selectbox(
            "Attendance Status",
            ["Present", "Absent"],
            key="attendance_status"
        )

        if st.button("Save Attendance"):
            attendance.setdefault(selected_id, {})
            attendance[selected_id][str(attendance_date)] = status

            attendance_file.write_text(
                json.dumps(attendance, indent=4),
                encoding="utf-8"
            )

            st.success("Attendance saved successfully!")
            st.rerun()

    else:
        selected_id = st.session_state.get("username", "")

    # Department-wise Attendance Report
    if is_admin:
        st.markdown("### 🏢 Department-wise Attendance Report")

        department_data = {}

        for sid, records in attendance.items():
            student = users.get(sid, {})
            department = student.get("department", "Unknown") or "Unknown"

            total_days = len(records)
            present_days = sum(
                1 for value in records.values()
                if value == "Present"
            )

            if department not in department_data:
                department_data[department] = {
                    "total": 0,
                    "present": 0
                }

            department_data[department]["total"] += total_days
            department_data[department]["present"] += present_days

        if department_data:
            report_rows = []

            for department, values in department_data.items():
                total_days = values["total"]
                present_days = values["present"]

                percentage = (
                    present_days / total_days * 100
                    if total_days else 0
                )

                report_rows.append({
                    "Department": department,
                    "Total Attendance": total_days,
                    "Present": present_days,
                    "Attendance %": round(percentage, 1)
                })

            department_df = pd.DataFrame(report_rows)

            # Department Attendance PDF Report
            pdf_buffer = io.BytesIO()
            pdf = canvas.Canvas(pdf_buffer)
            pdf.setTitle("Department Attendance Report")
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, 800, "Department-wise Attendance Report")

            pdf.setFont("Helvetica", 10)

            y = 760

            for _, row in department_df.iterrows():
                text = (
                    f"{row['Department']} - "
                    f"Total: {row['Total Attendance']}, "
                    f"Present: {row['Present']}, "
                    f"Attendance: {row['Attendance %']}%"
                )
                pdf.drawString(50, y, text)
                y -= 25

                if y < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = 800

            pdf.save()

            st.download_button(
                "⬇️ Download Department Report (PDF)",
                data=pdf_buffer.getvalue(),
                file_name="department_attendance_report.pdf",
                mime="application/pdf"
            )

            st.dataframe(
                department_df,
                use_container_width=True
            )

            st.bar_chart(
                department_df.set_index("Department")["Attendance %"]
            )
        else:
            st.info("No department attendance records found.")

    # Individual Attendance Summary
    student_attendance = attendance.get(selected_id, {})

    st.markdown("### 📊 Attendance Summary")

    total = len(student_attendance)

    present = sum(
        1 for value in student_attendance.values()
        if value == "Present"
    )

    absent = sum(
        1 for value in student_attendance.values()
        if value == "Absent"
    )

    percentage = (present / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Days", total)
    c2.metric("Present", present)
    c3.metric("Absent", absent)
    c4.metric("Attendance %", f"{percentage:.1f}%")

    # Attendance History
    st.markdown("### 📋 Attendance History")

    if student_attendance:
        history = pd.DataFrame(
            list(student_attendance.items()),
            columns=["Date", "Status"]
        )

        st.dataframe(history, use_container_width=True)

        csv = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Attendance Report (CSV)",
            data=csv,
            file_name="attendance_report.csv",
            mime="text/csv"
        )

        # PDF Attendance Report
        pdf_buffer = io.BytesIO()
        pdf = canvas.Canvas(pdf_buffer)

        pdf.setTitle(f"Attendance Report - {selected_id}")
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, 800, "Student Attendance Report")

        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, 760, f"Register Number: {selected_id}")
        pdf.drawString(50, 735, f"Total Days: {total}")
        pdf.drawString(50, 710, f"Present: {present}")
        pdf.drawString(50, 685, f"Absent: {absent}")
        pdf.drawString(50, 660, f"Attendance: {percentage:.1f}%")

        pdf.save()

        st.download_button(
            "⬇️ Download Attendance Report (PDF)",
            data=pdf_buffer.getvalue(),
            file_name=f"{selected_id}_attendance_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("No attendance records found.")

# ----------------------------------------------------------------------------
# Page 4: ML prediction
# ----------------------------------------------------------------------------
def page_prediction():
    st.markdown("## 🤖 ML Prediction")
    st.caption("Uses the models trained by `src/model.py` on this project's dataset.")

    try:
        bundle = get_bundle()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the trained model: {exc}")
        return

    enc = bundle["encoders"]
    rng = lambda c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))  # noqa: E731

    with st.form("predict_form"):
        model_key = st.selectbox(
            "Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k]
        )
        st.markdown("**Profile**")
        a1, a2, a3, a4 = st.columns(4)
        gender = a1.selectbox("Gender", list(enc["Gender"].classes_))
        branch = a2.selectbox("Branch", list(enc["Branch"].classes_))
        training = a3.selectbox("Placement training", list(enc["Placement_Training"].classes_))
        work = a4.selectbox("Work experience", list(enc["Work_Experience"].classes_))

        st.markdown("**Academics**")
        b1, b2, b3, b4 = st.columns(4)
        lo, hi, med = rng("CGPA")
        cgpa = b1.number_input("CGPA (0–10)", 0.0, 10.0, round(med, 2), 0.1)
        ssc = b2.number_input("SSC marks (%)", 0.0, 100.0, round(rng("SSC_Marks")[2], 1), 0.5)
        hsc = b3.number_input("HSC marks (%)", 0.0, 100.0, round(rng("HSC_Marks")[2], 1), 0.5)
        backlogs = b4.number_input("Backlogs", 0, 20, 0, 1)

        st.markdown("**Skills & experience**")
        c1, c2, c3, c4 = st.columns(4)
        iq = c1.number_input("IQ", 50.0, 200.0, round(rng("IQ")[2], 0), 1.0)
        aptitude = c2.number_input("Aptitude test score (0–100)", 0.0, 100.0, round(rng("Aptitude_Test_Score")[2], 1), 0.5)
        comm = c3.number_input("Communication skills (1–10)", 1.0, 10.0, round(rng("Communication_Skills")[2], 1), 0.1)
        extra = c4.number_input("Extra-curricular score (0–10)", 0.0, 10.0, round(rng("Extra_Curricular_Score")[2], 1), 0.1)

        d1, d2, d3, _ = st.columns(4)
        internships = d1.number_input("Internships", 0, 20, 1, 1)
        projects = d2.number_input("Projects", 0, 30, 2, 1)
        certs = d3.number_input("Certifications", 0, 30, 1, 1)

        submitted = st.form_submit_button("Predict placement")

    if not submitted:
        return

    values = {
        "Gender": gender, "Branch": branch, "Placement_Training": training, "Work_Experience": work,
        "SSC_Marks": ssc, "HSC_Marks": hsc, "CGPA": cgpa, "IQ": iq,
        "Internships": internships, "Projects": projects, "Certifications": certs,
        "Aptitude_Test_Score": aptitude, "Communication_Skills": comm,
        "Extra_Curricular_Score": extra, "Backlogs": backlogs,
    }

    try:
        res = predict_student(values, bundle, model_key)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    placed_flag = res["label"] == "Placed"
    css = "ok" if placed_flag else "bad"
    icon = "✅" if placed_flag else "⚠️"
    st.markdown(
        f'<div class="result {css}"><div class="verdict">{icon} {res["label"]}</div>'
        f'<div class="sub">Predicted probability of placement: <b>{res["prob_placed"] * 100:.1f}%</b> '
        f'· Model: {res["model"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(res["prob_placed"], 0.0), 1.0))
             # Skill Gap Analysis
    st.markdown("### 🎯 Skill Gap Analysis")
    st.caption("Suggestions to improve placement readiness.")

    skill_scores = {
        "CGPA": (cgpa / 10) * 100,
        "Aptitude": aptitude,
        "Communication": (comm / 10) * 100,
        "Extra-curricular": (extra / 10) * 100,
    }

    suggestions = []

    if cgpa < 7.0:
        suggestions.append("Improve CGPA through regular academic practice.")
    if aptitude < 60:
        suggestions.append("Practice aptitude and logical reasoning.")
    if comm < 6:
        suggestions.append("Improve communication skills.")
    if extra < 5:
        suggestions.append("Participate in extracurricular activities.")
    if internships < 1:
        suggestions.append("Complete at least one internship.")
    if projects < 2:
        suggestions.append("Build more practical projects.")
    if certs < 1:
        suggestions.append("Complete skill-based certifications.")
    if backlogs > 0:
        suggestions.append("Clear pending backlogs.")

    st.markdown("**Your Skill Scores**")
    for skill, score in skill_scores.items():
        st.write(f"{skill}: {score:.1f}%")
        st.progress(int(score))
    st.markdown("**Improvement Suggestions**")

    if suggestions:
        for suggestion in suggestions:
            st.warning("📌 " + suggestion)
    else:
        st.success("Great! Your entered scores meet the basic skill targets.")

    # Model Performance
    metrics = bundle.get("metrics", {}).get(res["model"])

    if metrics:
        st.caption(
            f"{res['model']} test-set performance on this dataset: accuracy "
            f"{metrics['accuracy'] * 100:.1f}%, ROC-AUC {metrics['roc_auc']:.3f}. "
            "This is a statistical estimate from a synthetic dataset, not a guarantee."
        )
# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
if page == PAGES[0]:
    page_home()
elif page == PAGES[1]:
    page_profile()
elif page == PAGES[2]:
    page_analysis()
elif page == PAGES[3]:
    page_students()
elif page == PAGES[4]:
    page_attendance()
else:
    page_prediction()