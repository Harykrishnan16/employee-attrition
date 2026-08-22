# Employee Attrition Prediction using Machine Learning

An end-to-end machine learning project that predicts **employee attrition** using employee demographic, job, satisfaction, income, and experience-related features.

The project covers the complete machine learning workflow, including **data preprocessing, exploratory data analysis (EDA), feature engineering, model training, model evaluation, model comparison, and saving the best-performing model for a Streamlit dashboard**.

## 📌 Project Overview

Employee attrition is an important challenge for organizations because high employee turnover can increase recruitment and training costs.

This project uses historical employee data to identify patterns associated with employee attrition and build classification models that predict whether an employee is likely to leave the organization.

### 🎯 Objective

* Analyze employee attrition patterns
* Clean and preprocess employee data
* Perform exploratory data analysis
* Engineer meaningful features
* Train multiple classification models
* Compare model performance
* Select the best-performing model based on **AUC-ROC**
* Save the trained pipeline for future predictions and Streamlit deployment

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical computations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine learning
* **Joblib** – Model serialization
* **Streamlit** – Interactive dashboard

---

## 🔄 Machine Learning Workflow

```text
Raw Employee Data
       ↓
Data Loading
       ↓
Data Cleaning & Preprocessing
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Train / Test Split
       ↓
Data Preprocessing Pipeline
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Best Model Selection
       ↓
Save Trained Pipeline
       ↓
Streamlit Dashboard
```

---

## 🧹 Data Preprocessing

The project performs several preprocessing steps:

* Removes constant and non-informative columns:

  * `EmployeeCount`
  * `StandardHours`
  * `Over18`
  * `EmployeeNumber`
* Handles missing numerical values using the **median**
* Handles missing categorical values using the **mode**
* Removes duplicate records
* Converts the target variable:

  * `Yes` → `1`
  * `No` → `0`

---

## ⚙️ Feature Engineering

Additional features are created to improve the predictive capability of the models:

### 1. Tenure Category

Employees are grouped according to their years at the company:

* New (0–2 years)
* Established (3–5 years)
* Experienced (6–10 years)
* Veteran (10+ years)

### 2. Engagement Score

A composite score is calculated using:

* Job Satisfaction
* Environment Satisfaction
* Relationship Satisfaction
* Work-Life Balance

### 3. Income Per Year of Experience

Calculates monthly income relative to total working experience.

### 4. Promotion Stagnation

Identifies employees who have spent **5 or more years since their last promotion**.

### 5. Frequent Job Changer

Identifies employees who have worked for **4 or more companies**.

---

## 📊 Exploratory Data Analysis

The project generates visualizations to understand employee attrition patterns, including:

* Attrition distribution
* Attrition by department
* Attrition by overtime
* Age distribution by attrition
* Monthly income vs. attrition
* Correlation heatmap
* Confusion matrices
* ROC curves
* Feature importance

All generated EDA charts are stored inside:

```text
outputs/eda/
```

---

## 🤖 Machine Learning Models

Three classification algorithms are trained and evaluated:

### Logistic Regression

Used as a baseline classification model.

### Decision Tree

Used to capture non-linear relationships between employee characteristics and attrition.

### Random Forest

An ensemble-based model used to improve predictive performance and provide feature importance.

The models use:

* StandardScaler for numerical features
* OneHotEncoder for categorical features
* Scikit-learn Pipeline and ColumnTransformer

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Confusion Matrix
* ROC Curve

## The models are compared based on **AUC-ROC**, and the highest-performing model is selected as the final model.

## 📂 Project Structure

```text
employee-attrition-prediction-ml/
│
├── Employee_Attrition.csv
├── employee_attrition_model.py
├── streamlit_app.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── eda/
    │   ├── attrition_distribution.png
    │   ├── attrition_by_department.png
    │   ├── attrition_by_overtime.png
    │   ├── age_distribution.png
    │   ├── income_vs_attrition.png
    │   ├── correlation_heatmap.png
    │   ├── feature_importance.png
    │   ├── confusion_matrix_*.png
    │   └── roc_curve_*.png
    │
    ├── models/
    │   ├── best_model.pkl
    │   ├── best_model_name.pkl
    │   └── feature_names.pkl
    │
    ├── cleaned_employee_data.csv
    └── model_comparison.csv
```

## The Python pipeline creates the `outputs` directories and saves the cleaned dataset, model comparison results, trained model, preprocessing information, and feature names.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/employee-attrition-prediction-ml.git
cd employee-attrition-prediction-ml
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

The core script specifies the required Python packages for running the model pipeline.

### 3. Add the Dataset

Place the dataset in the expected location:

```text
Employee_Attrition.csv
```

The current script expects the dataset path to be configured through `DATA_PATH`.

### 4. Train the Models

Run:

```bash
python employee_attrition_model.py
```

This will:

1. Load the dataset
2. Clean the data
3. Engineer features
4. Generate EDA visualizations
5. Train three ML models
6. Evaluate the models
7. Compare model performance
8. Select the best model
9. Save the trained pipeline

### 5. Launch the Streamlit Dashboard

After model training:

```bash
streamlit run streamlit_app.py
```

---

## 📦 Output Files

The project generates:

| File                        | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `best_model.pkl`            | Best-performing trained ML pipeline               |
| `best_model_name.pkl`       | Name of the selected model                        |
| `feature_names.pkl`         | Features used by the model                        |
| `model_comparison.csv`      | Performance comparison of trained models          |
| `cleaned_employee_data.csv` | Cleaned and feature-engineered dataset            |
| `outputs/eda/`              | Generated EDA and model evaluation visualizations |

---

## 💡 Key Skills Demonstrated

This project demonstrates practical skills in:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Data Preprocessing
* Classification
* Machine Learning Model Training
* Model Evaluation
* Model Comparison
* Feature Importance Analysis
* Pipeline Development
* Model Serialization
* Streamlit Deployment

---

## 🔮 Future Improvements

Potential improvements for this project include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* Handling class imbalance using additional techniques
* SHAP-based model explainability
* Interactive Streamlit visualizations
* Cloud deployment
* Automated model retraining
* Model monitoring

---
