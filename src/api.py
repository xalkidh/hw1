from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Φόρτωση μοντέλου και scaler
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Student Dropout Prediction API")

class StudentFeatures(BaseModel):
    Age: float
    Gender: str
    Family_Income: float
    Internet_Access: str
    Study_Hours_per_Day: float
    Attendance_Rate: float
    Assignment_Delay_Days: int
    Travel_Time_Minutes: float
    Part_Time_Job: str
    Scholarship: str
    Stress_Index: float
    GPA: float
    Semester_GPA: float
    CGPA: float
    Semester: str
    Department: str
    Parental_Education: str

@app.get("/")
def root():
    return {"message": "Student Dropout Prediction API is running."}

@app.post("/predict", response_model=None)
def predict(student: StudentFeatures):
    data = pd.DataFrame([student.dict()])

    # Feature Engineering
    data["study_attendance_ratio"] = data["Study_Hours_per_Day"] * data["Attendance_Rate"] / 100
    data["stress_gpa_ratio"] = data["Stress_Index"] / (data["GPA"] + 0.1)

    # One-Hot Encoding
    cat_features = ["Gender", "Internet_Access", "Part_Time_Job", 
                    "Scholarship", "Semester", "Department", "Parental_Education"]
    data = pd.get_dummies(data, columns=cat_features, drop_first=True)

    # Προσθήκη missing columns με 0
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    # Σωστή σειρά στηλών
    data = data[expected_cols]

    # Scaling
    data_scaled = scaler.transform(data)

    # Πρόβλεψη
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return JSONResponse(content={
        "prediction": int(prediction),
        "label": "Dropout" if prediction == 1 else "No Dropout",
        "probability": round(float(probability), 4)
    })