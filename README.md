# 🏨 Hotel Booking Cancellation Prediction

## 📌 Overview

This project uses machine learning to predict whether a hotel booking is likely to be **cancelled** based on historical booking information.

The dataset contains **119,390 hotel bookings** with information related to customer details, booking characteristics, stay duration, pricing, market segment, deposit type, previous cancellations, and special requests.

The project follows an end-to-end machine learning workflow including **data cleaning, exploratory data analysis, feature engineering, preprocessing, model comparison, evaluation, and feature importance analysis**.

---

## 🎯 Business Problem

Hotel cancellations can create uncertainty in room inventory and lead to potential revenue loss.

The objective of this project is to predict whether a booking will be cancelled and identify the factors that contribute most to cancellation behavior.

---

## 📊 Dataset

* **Records:** 119,390
* **Features:** 33
* **Target Variable:** `is_canceled`

Where:

* `0` → Booking was not cancelled
* `1` → Booking was cancelled

### Key Features

* Lead Time
* Hotel
* Deposit Type
* Market Segment
* Customer Type
* ADR
* Previous Cancellations
* Previous Bookings
* Stay Duration
* Special Requests
* Room Type
* Arrival Date
* Country

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train-Test Split
     ↓
Categorical Encoding
     ↓
Logistic Regression
     ↓
Random Forest
     ↓
Model Comparison
     ↓
Model Evaluation
     ↓
Feature Importance
```

---

## 🧹 Data Preprocessing

The following steps were performed:

* Checked and handled missing values
* Removed unnecessary columns
* Converted date-related variables
* Created meaningful features
* Split the data into training and testing sets
* Applied **One-Hot Encoding** to categorical variables
* Used `ColumnTransformer` for preprocessing

### Train-Test Split

| Dataset  | Records |
| -------- | ------: |
| Training |  95,512 |
| Testing  |  23,878 |

After preprocessing and One-Hot Encoding, the feature space increased to approximately **257 features**.

---

## 🤖 Models Used

Two classification algorithms were trained and compared:

🤖 **Model 1: Logistic Regression**

Logistic Regression was used as the baseline classification model for predicting whether a hotel booking would be cancelled.

Since the target variable is_canceled is binary, Logistic Regression is suitable for estimating the probability of cancellation.

**Preprocessing**
Before training Logistic Regression:

Numerical features were kept in numerical form
Categorical features were converted using One-Hot Encoding
ColumnTransformer was used to apply the preprocessing pipeline
The training and testing datasets were kept separate to avoid data leakage

After encoding the categorical variables, the feature space increased to approximately 257 features.

**Model Training**

The Logistic Regression model was trained on the processed training dataset and evaluated on the unseen test dataset.

The model was used as a baseline, allowing its performance to be compared against the Random Forest model.

**Why Logistic Regression?**

Logistic Regression was selected because:

It is well suited for binary classification
It provides probability-based predictions
It is relatively simple and interpretable
It provides a strong baseline for comparison with more complex models
Evaluation

**The Logistic Regression model was evaluated using:**

Accuracy
Confusion Matrix
ROC-AUC

The ROC-AUC score was particularly useful because it evaluates how well the model distinguishes between cancelled and non-cancelled bookings across different probability thresholds.

**🌲 Model 2: Random Forest**

Random Forest was then trained to capture non-linear relationships and interactions between the booking features.

Unlike Logistic Regression, which assumes a linear relationship between features and the log-odds of the target, Random Forest can capture more complex patterns in the data.

The Random Forest model performed better overall and was selected as the final model.

Final Performance
Accuracy: ~89%
ROC-AUC: ~0.954

The final Random Forest confusion matrix was:

[[14175   858]
 [ 1790  7055]]

The model also allowed us to analyze feature importance, helping identify the variables that contributed most to cancellation predictions.

---

## 📈 Model Comparison

The two models were evaluated using classification metrics, including **Accuracy and ROC-AUC**.

Random Forest achieved the stronger overall performance and was selected as the final model.

### Final Model Performance

* **Accuracy:** ~89%
* **ROC-AUC:** ~0.954

---

## 📊 Confusion Matrix

The final Random Forest model produced the following confusion matrix:

[[14175   858]
 [ 1790  7055]]

This allows us to analyze:

* True Negatives
* False Positives
* False Negatives
* True Positives

In particular, the model identified **7,055 cancelled bookings correctly**, while **1,790 cancelled bookings were incorrectly predicted as non-cancelled**.

---

## 📈 ROC-AUC

The final model achieved a **ROC-AUC of approximately 0.954**.

ROC-AUC was used to evaluate how effectively the model distinguishes between cancelled and non-cancelled bookings across different classification thresholds.

---

## 🔍 Feature Importance

Feature importance from the Random Forest model was analyzed to understand which variables had the greatest influence on predictions.

Some of the important features included:

* `lead_time`
* `adr`
* `deposit_type`
* `country`
* `arrival_date_day_of_month`
* `arrival_date_week_number`
* `total_of_special_requests`
* `total_stay`
* `market_segment`
* `previous_cancellations`
* `required_car_parking_spaces`

This helps connect the machine learning predictions to meaningful business factors.

---

## 💡 Key Takeaways

The project demonstrates how historical hotel booking data can be used to predict cancellation risk.

Factors such as **lead time, deposit type, pricing, market segment, previous cancellation history, stay duration, and special requests** provide valuable information for predicting booking cancellations.

The project also demonstrates the importance of comparing different machine learning approaches rather than relying on a single model.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn 
* Vs Code 

## 👨‍💻 Author

**Shubhankar Bhate**

Data Science / Machine Learning Project
