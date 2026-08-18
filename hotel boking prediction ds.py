import pandas as pd
df = pd.read_csv("C:/Users/LENOVO/Desktop/project/hotel_new.csv")

print(df.describe())

print(df.isnull().sum())
print(df.columns)
print(df.info())

df = df.drop(columns=["company", "agent"])
print (df)

df = df.drop(columns=["reservation_status", "reservation_status_date"])

print(df.columns)

print(df.isnull().sum().sort_values(ascending=False))

df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")

print(df.isnull().sum().sort_values(ascending=False))

print(df["is_canceled"].value_counts())
print(df["is_canceled"].value_counts(normalize=True) * 100)

#cancellation by hotel
print(pd.crosstab(
    df["hotel"],
    df["is_canceled"],
    normalize="index"
)* 100)

#cancellation by deposite type
print(pd.crosstab(
    df["deposit_type"],
    df["is_canceled"],
    normalize="index"
) * 100)

print(pd.crosstab(
    df["deposit_type"],
    df["market_segment"],
    normalize="index"
) * 100)

#cancellation rate by market segment 

print(pd.crosstab(
    df["market_segment"],
    df["is_canceled"],
    normalize="index"
) * 100)

#checking counts for each segment
print(df["market_segment"].value_counts())

#checking people who book in advance cancells more?
print(
df.groupby("is_canceled")["lead_time"].agg(
    ["count", "mean", "median", "min", "max"]
))

#cancellation rate by lead time groups : 1)creating bins and then will see 
df["lead_time_group"] = pd.cut(
    df["lead_time"],
    bins=[0, 30, 60, 90, 180, 365, float("inf")],
    labels=["0-30", "31-60", "61-90", "91-180", "181-365", "365+"]
)

print(df)

print(pd.crosstab(
    df["lead_time_group"],
    df["is_canceled"],
    normalize="index"
) * 100)


# do customers who have cancelled previouslt tend to cancel more ?
#1) creatin new feature : 
df["has_previous_cancellation"] = (
    df["previous_cancellations"] > 0
).astype(int)

print(pd.crosstab(
    df["has_previous_cancellation"],
    df["is_canceled"],
    normalize="index"
) * 100)


#wwe move toward prevvious not caancelled which have more succeful bookings previusly
df["has_previous_booking"] = (
    df["previous_bookings_not_canceled"] > 0
).astype(int)

print(pd.crosstab(
    df["has_previous_booking"],
    df["is_canceled"],
    normalize="index"
) * 100)

#If a booking has been changed at least once, does its cancellation rate differ?
# answer - Bookings that were modified at least once were less likely to be cancelled in this dataset.
df["has_booking_change"] = (
    df["booking_changes"] > 0
).astype(int)

print(pd.crosstab(
    df["has_booking_change"],
    df["is_canceled"],
    normalize="index"
) * 100)

#how many days ticket was in waiting list before getting confirmed: 
print(df.groupby("is_canceled")["days_in_waiting_list"].agg(
    ["count", "mean", "median", "min", "max"]
))
#Does simply being on the waiting list relate to cancellation?
df["has_waiting_list"] = (
    df["days_in_waiting_list"] > 0
).astype(int)

print(pd.crosstab(
    df["has_waiting_list"],
    df["is_canceled"],
    normalize="index"
) * 100)

#Are customers who make more special requests less likely to cancel?
print(df.groupby("is_canceled")["total_of_special_requests"].agg(
    ["count", "mean", "median", "min", "max"]
))

df["has_special_request"] = (
    df["total_of_special_requests"] > 0
).astype(int)

print(pd.crosstab(
    df["has_special_request"],
    df["is_canceled"],
    normalize="index"
) * 100)


#creating total stay 
df["total_stay"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)

print(df.groupby("is_canceled")["total_stay"].agg(
    ["count", "mean", "median", "min", "max"]
))


#ML PREdiction work starts: 

print(df.columns)


df = df.drop(columns=["lead_time_group"])

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

print(X.shape)
print(y.shape)

#train Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#checking the shape of the split 
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

#one hot encoding
#which are categorical columns 
print(X_train.select_dtypes(include="object").columns)

#counting unique values of that columns 
print(X_train.select_dtypes(include="object").nunique().sort_values())

categorical_cols = X_train.select_dtypes(include="object").columns
numerical_cols = X_train.select_dtypes(exclude="object").columns
print("Categorical:", len(categorical_cols))
print("Numerical:", len(numerical_cols))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_cols = X_train.select_dtypes(include="object").columns
numerical_cols = X_train.select_dtypes(exclude="object").columns

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder, categorical_cols)
    ],
    remainder="passthrough"
)


X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
print(X_train_encoded.shape)
print(X_test_encoded.shape)


#logistic regrssion : (accuracy was good but recall was the issue)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)

model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

#random forest now 

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_encoded, y_train)
rf_pred = rf_model.predict(X_test_encoded)
from sklearn.metrics import classification_report

print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

rf_prob = rf_model.predict_proba(X_test_encoded)[:, 1]

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, rf_prob)

print("ROC-AUC:", auc)

#Feature importance - why we are buliding the model 
importance = pd.Series(
    rf_model.feature_importances_,
    index=preprocessor.get_feature_names_out()
)

importance = importance.sort_values(ascending=False)

print(importance.head(20))