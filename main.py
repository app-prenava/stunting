import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stunting_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "stunting_threshold.pkl"

app = FastAPI(title="Stunting Risk Prediction API", version="1.0.0")

def load_artifact(path, default=None):
    if path.exists():
        return joblib.load(path)
    if default is not None:
        return default
    raise FileNotFoundError(f"Missing artifact: {path}")

try:
    model = load_artifact(MODEL_PATH)
    feature_columns = load_artifact(FEATURE_PATH)
    threshold = float(load_artifact(THRESHOLD_PATH, default=0.5))
except Exception as e:
    model = None
    feature_columns = []
    threshold = 0.5

if feature_columns:
    StuntingInput = create_model(
        "StuntingInput",
        **{feature: (float, ...) for feature in feature_columns}
    )
else:
    class StuntingInput(BaseModel):
        pass

@app.get("/")
def root():
    return {"message": "Stunting Risk Prediction API is running."}

@app.get("/health")
def health():
    if model is None or not feature_columns:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return {"status": "healthy"}

@app.get("/features")
def features():
    if not feature_columns:
        raise HTTPException(status_code=503, detail="Features not loaded.")
    return {"features": feature_columns}

@app.post("/predict")
def predict(data: StuntingInput):
    if model is None or not feature_columns:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])[feature_columns]

        probability = float(model.predict_proba(df)[0][1])
        prediction = int(probability >= threshold)

        return {
            "status": "success",
            "prediction": prediction,
            "risk_label": "High Risk" if prediction == 1 else "Low Risk",
            "probability_stunting": round(probability, 4),
            "threshold": threshold,
            "model_type": type(model).__name__,
            "note": "Early screening system, not a definitive diagnostic tool."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))