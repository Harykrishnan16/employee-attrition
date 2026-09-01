"""
Employee Attrition Dashboard (Streamlit)
=========================================
Run this AFTER employee_attrition_model.py has produced files in ./outputs/

    streamlit run streamlit_app.py

Provides:
    - Attrition trend overview (charts)
    - At-risk employee ranking (using the trained model on the dataset)
    - Single-employee "what-if" prediction form
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_employee_data.csv")

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    return model, feature_names, model_name


def check_artifacts_exist():
    missing = []
    if not os.path.exists(DATA_PATH):
        missing.append(DATA_PATH)
    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        missing.append(os.path.join(MODEL_DIR, "best_model.pkl"))
    if missing:
        st.error(
            "Required files not found: " + ", ".join(missing) +
            "\n\nPlease run `python employee_attrition_model.py` first to generate them."
        )
        st.stop()


check_artifacts_exist()
df = load_data()
model, feature_names, model_name = load_model()

st.title("👥 Employee Attrition Dashboard")
st.caption(f"Model in use: **{model_name}**")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🚨 At-Risk Employees", "🔮 Predict for an Employee"])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    total = len(df)
    left = int(df["Attrition"].sum())
    rate = left / total * 100

    col1.metric("Total Employees", total)
    col2.metric("Employees Who Left", left)
    col3.metric("Attrition Rate", f"{rate:.1f}%")

    st.subheader("Attrition by Department")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x="Department", hue="Attrition", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Attrition by OverTime")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x="OverTime", hue="Attrition", data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

    with colB:
        st.subheader("Monthly Income vs Attrition")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

    st.subheader("Job Satisfaction Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x="JobSatisfaction", hue="Attrition", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 2: At-risk employees ranking
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Employees Ranked by Predicted Attrition Risk")

    X_all = df[feature_names]
    probs = model.predict_proba(X_all)[:, 1]

    ranked = df.copy()
    ranked["AttritionRiskScore"] = probs
    ranked = ranked.sort_values("AttritionRiskScore", ascending=False)

    top_n = st.slider("Show top N at-risk employees", 5, 100, 20)
    display_cols = [
        "AttritionRiskScore", "Age", "Department", "JobRole", "MonthlyIncome",
        "JobSatisfaction", "OverTime", "YearsAtCompany", "Attrition",
    ]
    st.dataframe(
        ranked[display_cols].head(top_n).style.format({"AttritionRiskScore": "{:.1%}"}),
        use_container_width=True,
    )

    st.download_button(
        "Download full ranked list as CSV",
        ranked[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="at_risk_employees.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 3: Single-employee prediction (what-if form)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Predict Attrition Risk for a Single Employee")
    st.caption("Fill in employee details to estimate their probability of leaving.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 65, 30)
            monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000, step=500)
            job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=Very High)", [1, 2, 3, 4], index=2)
            env_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4], index=2)
            work_life_balance = st.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4], index=2)
            distance_from_home = st.number_input("Distance From Home (miles)", 0, 50, 5)

        with c2:
            department = st.selectbox("Department", df["Department"].unique())
            job_role = st.selectbox("Job Role", df["JobRole"].unique())
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            marital_status = st.selectbox("Marital Status", df["MaritalStatus"].unique())
            business_travel = st.selectbox("Business Travel", df["BusinessTravel"].unique())
            gender = st.selectbox("Gender", df["Gender"].unique())

        with c3:
            years_at_company = st.number_input("Years At Company", 0, 40, 5)
            total_working_years = st.number_input("Total Working Years", 0, 45, 8)
            num_companies_worked = st.number_input("Number of Companies Worked", 0, 10, 2)
            years_since_promotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
            percent_salary_hike = st.number_input("Percent Salary Hike", 0, 50, 15)
            stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        # Build a single-row DataFrame with defaults from the dataset median/mode
        # for any feature not explicitly captured in the form, then overwrite
        # with the values the user provided.
        row = {}
        for col in feature_names:
            if col in df.select_dtypes(include="number").columns:
                row[col] = df[col].median()
            else:
                row[col] = df[col].mode()[0]

        row.update({
            "Age": age,
            "MonthlyIncome": monthly_income,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": env_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "DistanceFromHome": distance_from_home,
            "Department": department,
            "JobRole": job_role,
            "OverTime": overtime,
            "MaritalStatus": marital_status,
            "BusinessTravel": business_travel,
            "Gender": gender,
            "YearsAtCompany": years_at_company,
            "TotalWorkingYears": total_working_years,
            "NumCompaniesWorked": num_companies_worked,
            "YearsSinceLastPromotion": years_since_promotion,
            "PercentSalaryHike": percent_salary_hike,
            "StockOptionLevel": stock_option_level,
            "EngagementScore": (job_satisfaction + env_satisfaction + work_life_balance + job_satisfaction) / 4,
            "PromotionStagnant": 1 if years_since_promotion >= 5 else 0,
            "FrequentJobChanger": 1 if num_companies_worked >= 4 else 0,
            "IncomePerYearExperience": monthly_income / max(total_working_years, 1),
        })

        # Recompute tenure category to match training logic
        if years_at_company <= 2:
            tenure_cat = "New (0-2y)"
        elif years_at_company <= 5:
            tenure_cat = "Established (3-5y)"
        elif years_at_company <= 10:
            tenure_cat = "Experienced (6-10y)"
        else:
            tenure_cat = "Veteran (10y+)"
        row["TenureCategory"] = tenure_cat

        input_df = pd.DataFrame([row])[feature_names]
        risk = model.predict_proba(input_df)[0, 1]

        st.metric("Predicted Attrition Risk", f"{risk:.1%}")
        if risk >= 0.5:
            st.error("⚠️ High risk of attrition — consider a retention conversation.")
        elif risk >= 0.3:
            st.warning("⚠️ Moderate risk — keep an eye on engagement.")
        else:
            st.success("✅ Low risk of attrition.")
