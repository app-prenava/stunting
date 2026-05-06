import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stunting_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

def test_inference():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    
    print(f"Model type: {type(model)}")
    print(f"Features: {features}")
    
    FEATURE_ORDER = [
        "child_gender", "mother_education_level", "mother_employment_status",
        "mother_height_cm", "improved_water", "improved_sanitation",
        "home_ownership", "has_electricity", "has_refrigerator",
        "has_tv", "mother_age_at_birth", "is_teenage_mother",
        "is_high_risk_mother_age", "has_delivery_insurance",
        "anc_clinic_midwife", "anc_hospital", "anc_traditional_other",
        "anc_unknown"
    ]
    
    payload = {
        "child_gender": 1, "mother_education_level": 4, "mother_employment_status": 1,
        "mother_height_cm": 155.0, "improved_water": 1, "improved_sanitation": 1,
        "home_ownership": 1, "has_electricity": 1, "has_refrigerator": 0,
        "has_tv": 1, "mother_age_at_birth": 24.0, "is_teenage_mother": 0,
        "is_high_risk_mother_age": 0, "has_delivery_insurance": 1,
        "anc_clinic_midwife": 0, "anc_hospital": 0, "anc_traditional_other": 1,
        "anc_unknown": 0
    }
    
    print("\nBuilding input array...")
    input_list = [float(payload[f]) for f in FEATURE_ORDER]
    input_array = np.array([input_list], dtype=np.float32)
    
    print(f"Input array shape: {input_array.shape}")
    print(f"Input array: {input_array}")
    
    print("\nRunning model.predict...")
    try:
        pred = model.predict(input_array)
        print(f"Prediction: {pred}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        
    print("\nRunning model.predict_proba...")
    try:
        proba = model.predict_proba(input_array)
        print(f"Probability: {proba}")
    except Exception as e:
        print(f"Probability failed: {e}")

if __name__ == "__main__":
    test_inference()
