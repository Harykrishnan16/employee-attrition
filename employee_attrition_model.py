"""
Employee Attrition Prediction
==============================
End-to-end pipeline: data preprocessing -> EDA -> feature engineering ->
model training (Logistic Regression, Decision Tree, Random Forest) ->
evaluation -> saving the best model for the Streamlit dashboard.

How to run (in VS Code terminal):
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    python employee_attrition_model.py

Expected input file (same folder as this script, or update DATA_PATH below):
    Employee_Attrition.csv

Outputs created:
    outputs/eda/*.png                  -> EDA charts
    outputs/models/best_model.pkl      -> trained model (joblib)
    outputs/models/preprocessor.pkl    -> fitted preprocessing pipeline
    outputs/models/feature_names.pkl   -> feature names used by the model
    outputs/model_comparison.csv       -> metrics for all trained models
    outputs/cleaned_employee_data.csv  -> cleaned dataset (project deliverable)
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # safe backend for headless / VS Code runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "../ Employee/Employee_Attrition.csv"
OUTPUT_DIR = "outputs"
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RANDOM_STATE = 42

# Columns that are constant / non-informative for modeling
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

sns.set_theme(style="whitegrid")


def ensure_dirs():
    for d in (OUTPUT_DIR, EDA_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop constant / non-informative columns that leak no signal
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped non-informative columns: {cols_to_drop}")

    # Handle missing values (numeric -> median, categorical -> mode)
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    # Drop exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - len(df)} duplicate rows")

    # Standardize target column
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure category buckets
    df["TenureCategory"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 2, 5, 10, 100],
        labels=["New (0-2y)", "Established (3-5y)", "Experienced (6-10y)", "Veteran (10y+)"],
    )

    # Composite engagement score from satisfaction-related fields
    df["EngagementScore"] = (
        df["JobSatisfaction"]
        + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"]
        + df["WorkLifeBalance"]
    ) / 4

    # Income per year of experience (avoid div-by-zero)
    df["IncomePerYearExperience"] = df["MonthlyIncome"] / df["TotalWorkingYears"].replace(0, 1)

    # Promotion stagnation flag
    df["PromotionStagnant"] = (df["YearsSinceLastPromotion"] >= 5).astype(int)

    # Job hopping indicator
    df["FrequentJobChanger"] = (df["NumCompaniesWorked"] >= 4).astype(int)

    print("Engineered features: TenureCategory, EngagementScore, "
          "IncomePerYearExperience, PromotionStagnant, FrequentJobChanger")
    return df


# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("\nRunning EDA, saving charts to:", EDA_DIR)

    # Attrition distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Attrition", data=df, palette="Set2")
    plt.title("Attrition Distribution (0=Stayed, 1=Left)")
    plt.savefig(os.path.join(EDA_DIR, "attrition_distribution.png"), bbox_inches="tight")
    plt.close()

    # Attrition by department
    plt.figure(figsize=(7, 4))
    sns.countplot(x="Department", hue="Attrition", data=df, palette="Set2")
    plt.title("Attrition by Department")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(EDA_DIR, "attrition_by_department.png"), bbox_inches="tight")
    plt.close()

    # Attrition by overtime
    plt.figure(figsize=(5, 4))
    sns.countplot(x="OverTime", hue="Attrition", data=df, palette="Set2")
    plt.title("Attrition by OverTime")
    plt.savefig(os.path.join(EDA_DIR, "attrition_by_overtime.png"), bbox_inches="tight")
    plt.close()

    # Age distribution by attrition
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="Age", hue="Attrition", fill=True, common_norm=False, alpha=0.4)
    plt.title("Age Distribution by Attrition")
    plt.savefig(os.path.join(EDA_DIR, "age_distribution.png"), bbox_inches="tight")
    plt.close()

    # Monthly income vs attrition
    plt.figure(figsize=(5, 4))
    sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette="Set2")
    plt.title("Monthly Income vs Attrition")
    plt.savefig(os.path.join(EDA_DIR, "income_vs_attrition.png"), bbox_inches="tight")
    plt.close()

    # Correlation heatmap (numeric features)
    plt.figure(figsize=(14, 10))
    num_df = df.select_dtypes(include=np.number)
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    print("EDA charts saved.")


# ---------------------------------------------------------------------------
# 4. Build preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
        ]
    )
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    return preprocessor


# ---------------------------------------------------------------------------
# 5. Train & evaluate models
# ---------------------------------------------------------------------------
def evaluate_model(name, y_test, y_pred, y_proba):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))
    return metrics


def plot_confusion_and_roc(name, y_test, y_pred, y_proba):
    safe_name = name.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.savefig(os.path.join(EDA_DIR, f"confusion_matrix_{safe_name}.png"), bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {name}")
    ax.legend()
    fig.savefig(os.path.join(EDA_DIR, f"roc_curve_{safe_name}.png"), bbox_inches="tight")
    plt.close(fig)


def train_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(name, y_test, y_pred, y_proba)
        plot_confusion_and_roc(name, y_test, y_pred, y_proba)

        results.append(metrics)
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
    return results_df, fitted_pipelines


# ---------------------------------------------------------------------------
# 6. Feature importance (for tree-based best model)
# ---------------------------------------------------------------------------
def plot_feature_importance(pipe, model_name):
    classifier = pipe.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return

    preprocessor = pipe.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=imp_df, palette="viridis")
    plt.title(f"Top 15 Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "feature_importance.png"), bbox_inches="tight")
    plt.close()
    print("Saved feature importance chart.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    # Save cleaned dataset (project deliverable)
    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_employee_data.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")

    run_eda(df)

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor(X)
    results_df, fitted_pipelines = train_models(X_train, X_test, y_train, y_test, preprocessor)

    print("\n=== Model Comparison (sorted by AUC-ROC) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    # Pick best model by AUC-ROC
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\nBest model: {best_model_name}")

    plot_feature_importance(best_pipeline, best_model_name)

    # Save the full pipeline (preprocessing + model) for the Streamlit app
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(best_model_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))
    print(f"\nSaved trained pipeline to {os.path.join(MODEL_DIR, 'best_model.pkl')}")
    print("Done. Run 'streamlit run streamlit_app.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
