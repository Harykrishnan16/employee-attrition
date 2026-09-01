# 👥 Employee Attrition Prediction & Risk Dashboard

An end-to-end **Machine Learning project for predicting employee attrition** and identifying employees who may be at higher risk of leaving an organization.

This project combines **data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model comparison, model evaluation, and an interactive Streamlit dashboard**.

The trained model can be used to estimate employee attrition risk and support data-driven employee retention strategies.

---

## 📌 Project Overview

Employee attrition can have a significant impact on organizations through increased recruitment costs, loss of experienced employees, and reduced productivity.

This project uses employee-related information such as:

* Age
* Monthly Income
* Job Satisfaction
* Environment Satisfaction
* Work-Life Balance
* Overtime
* Department
* Job Role
* Years at Company
* Total Working Years
* Number of Companies Worked
* Years Since Last Promotion
* Business Travel
* Marital Status
* Stock Option Level
* and other employee attributes

to build a machine learning system that predicts the probability of employee attrition.

---

## 🎯 Objectives

The main objectives of this project are:

1. Clean and preprocess employee data.
2. Perform exploratory data analysis to identify attrition patterns.
3. Engineer meaningful features related to employee engagement and career progression.
4. Train multiple classification models.
5. Compare model performance using appropriate evaluation metrics.
6. Select the best-performing model based on **AUC-ROC**.
7. Save the trained model and preprocessing pipeline.
8. Build an interactive Streamlit dashboard.
9. Rank employees based on predicted attrition risk.
10. Provide an individual employee **what-if prediction** interface.

---

## 🔄 Project Workflow

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
Feature Scaling & Encoding
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Best Model Selection
       ↓
Model Serialization
       ↓
Streamlit Dashboard
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Removing non-informative columns
* Handling missing numerical values using the median
* Handling missing categorical values using the mode
* Removing duplicate records
* Converting the `Attrition` target variable:

  * `Yes → 1`
  * `No → 0`

The project removes columns such as `EmployeeCount`, `StandardHours`, `Over18`, and `EmployeeNumber` because they are not useful for predictive modeling.

---

## 🛠️ Feature Engineering

Several additional features are created to improve the model's ability to capture employee behavior and career patterns.

### 1. Tenure Category

Employees are grouped based on their years at the company:

* `New (0-2y)`
* `Established (3-5y)`
* `Experienced (6-10y)`
* `Veteran (10y+)`

### 2. Engagement Score

A composite score is created using:

* Job Satisfaction
* Environment Satisfaction
* Relationship Satisfaction
* Work-Life Balance

### 3. Income Per Year of Experience

```text
Monthly Income / Total Working Years
```

### 4. Promotion Stagnation

Employees with five or more years since their last promotion are flagged as potentially experiencing promotion stagnation.

### 5. Frequent Job Changer

Employees who have worked at four or more companies are flagged as frequent job changers.

---

## 📊 Exploratory Data Analysis

The project generates visualizations to investigate relationships between employee characteristics and attrition.

### Key analyses include:

* Attrition Distribution
* Attrition by Department
* Attrition by Overtime
* Age Distribution by Attrition
* Monthly Income vs Attrition
* Correlation Heatmap
* Confusion Matrices
* ROC Curves
* Feature Importance

The EDA charts are automatically saved under:

```text
outputs/eda/
```

---

## 🤖 Machine Learning Models

Three classification algorithms are trained and compared:

### 1. Logistic Regression

Used as a linear baseline classification model.

### 2. Decision Tree

A tree-based model capable of capturing non-linear relationships between employee characteristics and attrition.

### 3. Random Forest

An ensemble of decision trees designed to improve predictive performance and robustness.

The implementation uses class balancing to address the difference between employees who stayed and employees who left.

---

## 📈 Model Evaluation

The models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **AUC-ROC**

The model comparison is sorted by **AUC-ROC**, and the highest-performing model is selected as the final model.

---

## 💾 Model Artifacts

After training, the project generates:

```text
outputs/
│
├── cleaned_employee_data.csv
├── model_comparison.csv
│
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
└── models/
    ├── best_model.pkl
    ├── feature_names.pkl
    └── best_model_name.pkl
```

The complete preprocessing + model pipeline is saved as `best_model.pkl` so that the same transformations used during training can be applied when making predictions in the dashboard.

---

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard with three main sections.

### 1. 📊 Overview

Provides a high-level view of:

* Total employees
* Employees who left
* Overall attrition rate
* Attrition by department
* Attrition by overtime
* Monthly income vs attrition
* Job satisfaction distribution

### 2. 🚨 At-Risk Employees

Employees are ranked according to their predicted probability of attrition.

The dashboard displays:

* Attrition Risk Score
* Age
* Department
* Job Role
* Monthly Income
* Job Satisfaction
* Overtime
* Years at Company
* Actual Attrition

Users can also download the ranked employee list as a CSV file.

### 3. 🔮 Individual Employee Prediction

The dashboard provides a **what-if prediction form** where users can enter employee characteristics and estimate their probability of attrition.

The prediction is categorized as:

```text
Risk < 30%       → Low Risk
30% - 49.9%      → Moderate Risk
≥ 50%            → High Risk
```

---

## 🧰 Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming language      |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| Matplotlib   | Data visualization        |
| Seaborn      | Statistical visualization |
| Scikit-learn | Machine learning          |
| Joblib       | Model serialization       |
| Streamlit    | Interactive dashboard     |
| Git & GitHub | Version control           |

---

## 📁 Project Structure

```text
employee-attrition-prediction/
│
├── employee_attrition_model.py
├── streamlit_app.py
├── Employee-Attrition_-_Employee-Attrition.csv
├── README.md
├── requirements.txt
│
└── outputs/
    ├── cleaned_employee_data.csv
    ├── model_comparison.csv
    │
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
    └── models/
        ├── best_model.pkl
        ├── feature_names.pkl
        └── best_model_name.pkl
```

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

Or:

```bash
pip install -r requirements.txt
```

### Step 3: Train the Models

Make sure the employee dataset is available in the expected location.

Run:

```bash
python employee_attrition_model.py
```

This will:

* Load the dataset
* Clean the data
* Engineer features
* Generate EDA visualizations
* Train three ML models
* Evaluate the models
* Select the best model
* Save the model artifacts

### Step 4: Launch the Dashboard

After model training is complete:

```bash
streamlit run streamlit_app.py
```

---

## 📌 Key Project Highlights

* ✅ End-to-end machine learning workflow
* ✅ Data cleaning and preprocessing
* ✅ Feature engineering
* ✅ Exploratory data analysis
* ✅ Multiple classification algorithms
* ✅ Class-balanced model training
* ✅ Model comparison using AUC-ROC
* ✅ Confusion matrix and ROC curve analysis
* ✅ Feature importance analysis
* ✅ Saved ML pipeline using Joblib
* ✅ Interactive Streamlit dashboard
* ✅ Employee attrition risk ranking
* ✅ Individual employee risk prediction
* ✅ Downloadable at-risk employee report

---

## 💡 Business Use Case

This project demonstrates how machine learning can support **HR analytics and employee retention**.

Organizations could use an attrition-risk system to:

* Identify employees who may require additional engagement
* Understand factors associated with employee turnover
* Prioritize retention initiatives
* Support HR teams with data-driven insights
* Explore potential relationships between compensation, satisfaction, overtime, tenure, and attrition

> **Note:** Predicted attrition risk should be treated as a decision-support signal rather than a definitive statement about whether an individual employee will leave.

---

## 🔮 Future Improvements

Potential improvements to the project include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation for more robust model evaluation
* SHAP-based model explainability
* Interactive feature-importance analysis
* Additional HR analytics visualizations
* Model performance monitoring
* Deployment using Streamlit Cloud or another cloud platform
* Automated model retraining pipeline
* More sophisticated risk segmentation
