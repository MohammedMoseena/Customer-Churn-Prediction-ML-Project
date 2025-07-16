# 💼 Customer Churn Prediction – End-to-End ML Project

### 📊 Objective
To build a machine learning pipeline to predict telecom customer churn using a real-world dataset. The project aims to identify customers likely to cancel services and help the business take proactive actions to retain them, ultimately saving revenue.
---

## 📌 Table of Contents

* [Business Understanding](#business-understanding)
* [Technical Requirements](#technical-requirements)
* [Evaluation Metrics](#evaluation-metrics)
* [Data Understanding](#data-understanding)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
* [Modeling](#modeling)
* [Model Evaluation](#model-evaluation)
* [Revenue Impact](#revenue-impact)
* [Deployment & Inference](#deployment--inference)
* [Results](#results)
* [Tools & Libraries](#tools--libraries)

---

## 💡 Business Understanding

### ➤ Why Churn Prediction Matters

* **For Business:**

  * Reduce revenue loss
  * Improve customer experience
  * Enable targeted retention strategies

* **For Customers:**

  * Personalized offers
  * Less spam
  * Better support

* **For Internal Teams:**

  * Reduced workload
  * Prioritized customer engagement

### ➤ Churn Criteria

* **Churn = Yes** → Customer left/canceled service
* **Churn = No** → Customer retained service

### ➤ Misclassification Impact

* **False Negatives (Missed churners)** → Revenue loss
* **False Positives** → Increased retention cost

---

## ⚙️ Technical Requirements

* **High Accuracy** ✅
* **High Throughput (Scalability)** ✅
* **Interpretability** – Nice to have
* **Low Latency** – Not required (batch predictions)

---

## 📈 Evaluation Metrics

Due to class imbalance, **F1-Score** and **Recall** are prioritized over Accuracy.
Other metrics used:
* Precision
* AUC-ROC
* Confusion Matrix

---

## 🧠 Data Understanding

* **Dataset:** `Telco-Customer-Churn.csv`
* **Samples:** 7043 rows, 21 columns
* **Target Variable:** `Churn` (Yes/No)
* **Numerical Features:** Tenure, MonthlyCharges, SeniorCitizen
* **Categorical Features:** Gender, Contract Type, Internet Service, etc.

---

## 🔎 Exploratory Data Analysis (EDA)

* Target class (`Churn`) is **imbalanced**
* Categorical features analyzed using **bar charts**
* No missing values and outliers in the dataset
* Converted `TotalCharges` from object to float
* High correlation observed between `TotalCharges`, `tenure`, and `MonthlyCharges`

---

## 🧹 Data Preprocessing & Feature Engineering

* Dropped uninformative ID column
* Standardized numerical features using **StandardScaler**
* Applied **One-Hot Encoding** for categorical variables
* Saved fitted scalers and encoders using `pickle`

---

## 🤖 Modeling

* Handled class imbalance using **SMOTE + Undersampling**
* Trained and evaluated multiple models:

  * ✅ **Random Forest** (Best model)
* Applied **RandomizedSearchCV** for hyperparameter tuning on Random Forest

---

## 🧪 Model Evaluation

| Metric    | Before Tuning | After Tuning |
| --------- | ------------- | ------------ |
| F1 Score  | 0.571         | 0.598        |
| Recall    | 0.5855        | 0.6550       |
| Precision | 0.5586        | 0.5505       |
| AUC-ROC   | 0.8178        | 0.8291       |
| Accuracy  | 0.7672        | 0.7665       |

> ✅ **F1 and Recall Improved after tuning** → Fewer False Negatives (more churners caught)

---

## 💰 Revenue Impact

Estimated Revenue Saved by retaining correctly predicted churners:

* **Before Tuning:** ₹199,963.10
* **After Tuning:** ₹237,836.60 ✅

---

## 🚀 Deployment & Inference

* Saved model and scalers (`customer_churn_model.pkl`, `scaler.pkl`, `encoders.pkl`)
* Built **inference pipeline** to:

  * Accept raw customer input
  * Preprocess & encode
  * Predict churn & return probability

---

## ✅ Results

* Built full end-to-end churn prediction pipeline from data to deployment
* Aligned model performance with business KPIs
* Demonstrated measurable impact by estimating revenue saved
* Code modularized for reusability and real-world integration

---

## 🛠️ Tools & Libraries

* Python, Pandas, NumPy
* Matplotlib, Seaborn (EDA)
* Scikit-learn (Modeling, Preprocessing)
* Imbalanced-learn (SMOTE + Undersampling)
* XGBoost
* Pickle (Serialization)

---

## 📎 Project Structure (if applicable)

```
├── Customer-Churn-Prediction.ipynb
├── scaler.pkl
├── encoders.pkl
├── customer_churn_model.pkl
├── Telco-Customer-Churn.csv
```

---
