# Employee Attrition Prediction — Setup

## 1. Install dependencies
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

## 2. Folder setup
Put these three files in the same folder:
- `employee_attrition_model.py`
- `streamlit_app.py`
- `Employee-Attrition_-_Employee-Attrition.csv` (your dataset)

## 3. Train the model
```
python employee_attrition_model.py
```
This cleans the data, runs EDA (saves charts to `outputs/eda/`), engineers
features, trains Logistic Regression / Decision Tree / Random Forest,
prints/saves metrics (`outputs/model_comparison.csv`), and saves the best
model to `outputs/models/best_model.pkl`.

## 4. Launch the dashboard
```
streamlit run streamlit_app.py
```
Opens a browser dashboard with attrition overview charts, a ranked
at-risk-employee list, and a form to predict risk for a single employee.

## Alternative: SQL-style dashboard (`attrition_dashboard.py`)

This mirrors the architecture of `securecheck.py`: a single-file Streamlit
app that queries a PostgreSQL table named `employees` when available, and
automatically falls back to the CSV (`Employee-Attrition_-_Employee-Attrition.csv`)
when no database is configured. It includes:

- Overall KPI metrics (total employees, attrition count, overtime count)
- Sidebar quick filters (Department, Gender, Attrition)
- An "Advanced Search & Insights" expander with more filters (job role,
  marital status, overtime, age/income/tenure ranges) plus auto-generated
  insights and a bar chart
- A "Data Analysis" section (sample rows, column overview, data types)
- An "Advanced SQL Analytics" section with 6 prewritten SQL queries
  (subqueries + window functions) that each also have a pandas equivalent,
  so the dashboard works with or without Postgres
- A "Delete Employee Record" section (requires a live DB connection)

To use it with Postgres, update `DB_CONFIG` at the top of the file and load
the CSV into a table called `employees`. Otherwise it just runs on the CSV
automatically:
```
streamlit run attrition_dashboard.py
```

Note: this file focuses on descriptive/SQL analytics (matching
`securecheck.py`'s style) rather than the ML prediction. If you want the
attrition-risk model included in this same file, let me know and I'll merge
in a "Predict Attrition Risk" tab using the trained model from
`employee_attrition_model.py`.
