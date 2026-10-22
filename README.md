# Student Placement Analysis Dashboard

A Streamlit website built on the existing *Student Placement Data Analysis* Python project
(800 students: 589 placed, 211 not placed).

## Pages
1. **Home Dashboard** – metric cards, branch placement rate, top-5 features, model results
2. **Data Analysis** – all 11 charts from the `charts/` folder
3. **Student Data** – searchable table, branch / status / CGPA filters, summary statistics, CSV download
4. **ML Prediction** – enter student details and predict Placed / Not Placed with the trained model

## Run in VS Code (Windows)
1. Install Python 3.10+ from python.org (tick **Add Python to PATH**).
2. In VS Code: **File > Open Folder** and choose this folder.
3. Open the terminal (**Terminal > New Terminal**) and run:
   ```
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```
4. Open **http://localhost:8501** in Chrome (it usually opens automatically).
5. Stop the server with `Ctrl + C` in the terminal.

## Project structure
```
app.py                  Streamlit dashboard
requirements.txt
src/generate_data.py    creates data/placement_data.csv
src/eda_analysis.py     creates the EDA charts (01-08)
src/model.py            trains models, creates charts 09-11, saves models/placement_model.joblib
src/predict.py          prediction helper used by the app
data/                   placement_data.csv, model_results.csv, feature_correlation_with_placement.csv
charts/                 11 PNG charts
models/                 placement_model.joblib (trained models)
assets/                 project report (.docx)
.streamlit/config.toml  dark theme
```

## Re-training (optional)
```
python src/model.py
```
This regenerates the model file, charts 09-11 and `model_results.csv`.

## Notes
- The dataset is **synthetic** (created by `src/generate_data.py`), so results illustrate the method and are not real placement statistics.
- If the app says the model file is missing, run `python src/model.py` once.
- scikit-learn version differences can break loading the `.joblib` file; if so, run `python src/model.py` again.
