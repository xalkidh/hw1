import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path)
    # Αφαιρώ την στήλη Student_ID αφού δεν είναι χρήσιμη
    df = df.drop(columns=["Student_ID"], errors="ignore")
    return df

def feature_engineering(df):
    # Feature 1: study_attendance_ratio
    df["study_attendance_ratio"] = df["Study_Hours_per_Day"] * df["Attendance_Rate"] / 100

    # Feature 2: stress_gpa_ratio
    df["stress_gpa_ratio"] = df["Stress_Index"] / (df["GPA"] + 0.1)

    return df

def split_data(df):
    X = df.drop(columns=["Dropout"])
    y = df["Dropout"]

    # Πρώτο split: 90% train+val, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Δεύτερο split: 80% train, 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_feature_types(X_train):
    num_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    return num_features, cat_features

def handle_missing_values(X_train, X_val, X_test, num_features, cat_features):
    # Numerical: median
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_features] = num_imputer.fit_transform(X_train[num_features])
    X_val[num_features] = num_imputer.transform(X_val[num_features])
    X_test[num_features] = num_imputer.transform(X_test[num_features])

    # Categorical: most_frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
    X_val[cat_features] = cat_imputer.transform(X_val[cat_features])
    X_test[cat_features] = cat_imputer.transform(X_test[cat_features])

    return X_train, X_val, X_test

def handle_outliers(X_train, X_val, X_test, num_features):
    # IQR method (υπολογισμός στο train set)
    bounds = {}
    for col in num_features:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)

        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    return X_train, X_val, X_test, bounds

def encode_categoricals(X_train, X_val, X_test, cat_features):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    
    encoded_train = encoder.fit_transform(X_train[cat_features])
    encoded_val = encoder.transform(X_val[cat_features])
    encoded_test = encoder.transform(X_test[cat_features])

    encoded_cols = encoder.get_feature_names_out(cat_features)

    # Αντικατάσταση κατηγορικών με encoded
    X_train = X_train.drop(columns=cat_features).reset_index(drop=True)
    X_val = X_val.drop(columns=cat_features).reset_index(drop=True)
    X_test = X_test.drop(columns=cat_features).reset_index(drop=True)

    X_train = pd.concat([X_train, pd.DataFrame(encoded_train, columns=encoded_cols)], axis=1)
    X_val = pd.concat([X_val, pd.DataFrame(encoded_val, columns=encoded_cols)], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(encoded_test, columns=encoded_cols)], axis=1)

    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Αποθήκευση scaler
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def preprocess(path="data/dataset.csv"):
    df = load_data(path)
    df = feature_engineering(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    num_features, cat_features = get_feature_types(X_train)
    
    X_train, X_val, X_test = handle_missing_values(X_train, X_val, X_test, num_features, cat_features)
    X_train, X_val, X_test, bounds = handle_outliers(X_train, X_val, X_test, num_features)
    X_train, X_val, X_test = encode_categoricals(X_train, X_val, X_test, cat_features)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Target distribution - Train: {y_train.mean():.2%} dropout")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test