"""
Prenava - Stunting Risk Prediction API
Production-optimized FastAPI service.
- Model & SHAP loaded ONCE at startup
- Top-5 SHAP factors returned (lightweight payload)
- Inference + SHAP timing logged per request
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stunting_api")

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "stunting_model.pkl"
FEATURE_PATH    = BASE_DIR / "models" / "feature_columns.pkl"
THRESHOLD_PATH  = BASE_DIR / "models" / "stunting_threshold.pkl"
IMPUTER_PATH    = BASE_DIR / "models" / "stunting_imputer.pkl"
SCALER_PATH     = BASE_DIR / "models" / "stunting_scaler.pkl"

# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
model: object = None
feature_columns: list = []
threshold: float = 0.5
explainer: object = None
imputer: object = None
scaler: object = None
# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class StuntingInput(BaseModel):
    child_gender: int
    mother_education_level: int
    mother_employment_status: int
    mother_height_cm: float
    improved_water: int
    improved_sanitation: int
    home_ownership: int
    has_electricity: int
    has_refrigerator: int
    has_tv: int
    mother_age_at_birth: float
    is_teenage_mother: int
    is_high_risk_mother_age: int
    has_delivery_insurance: int
    anc_clinic_midwife: int
    anc_hospital: int
    anc_traditional_other: int
    anc_unknown: int


def _load_artifact(path: Path, default=None):
    if path.exists():
        return joblib.load(path)
    if default is not None:
        return default
    raise FileNotFoundError(f"Missing artifact: {path}")


# Linear model classes that need LinearExplainer
_LINEAR_MODEL_TYPES = (
    "LogisticRegression",
    "LinearSVC",
    "LinearSVR",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SGDClassifier",
    "SGDRegressor",
    "BayesianRidge",
    "ARDRegression",
)

# Tree-based model classes that need TreeExplainer
_TREE_MODEL_TYPES = (
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
)


def _build_explainer(fitted_model: object, columns: list) -> object:
    """
    Auto-detect and build the correct SHAP explainer for the fitted model.

    Priority:
      1. LinearExplainer  — for linear/GLM models (fast, exact)
      2. TreeExplainer    — for tree-based models (fast, exact)
      3. KernelExplainer  — universal fallback (slow, approximate)
    """
    model_name = type(fitted_model).__name__

    if model_name in _LINEAR_MODEL_TYPES:
        logger.info("Using shap.LinearExplainer for %s", model_name)
        # LinearExplainer needs a masker — use the feature distribution (interventional)
        import numpy as np
        # Use a zero-vector background as a simple but valid masker
        background = np.zeros((1, len(columns)))
        return shap.LinearExplainer(fitted_model, background)

    if model_name in _TREE_MODEL_TYPES:
        logger.info("Using shap.TreeExplainer for %s", model_name)
        return shap.TreeExplainer(fitted_model)

    # Unknown model type — use KernelExplainer with a minimal background
    logger.warning(
        "Unknown model type '%s'. Falling back to KernelExplainer (slow). "
        "Add it to _LINEAR_MODEL_TYPES or _TREE_MODEL_TYPES if inference is too slow.",
        model_name,
    )
    import numpy as np
    background = shap.maskers.Independent(np.zeros((1, len(columns))), max_samples=1)
    return shap.KernelExplainer(fitted_model.predict_proba, background)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + SHAP explainer ONCE at startup."""
    global model, feature_columns, threshold, explainer, imputer, scaler

    t0 = time.perf_counter()
    logger.info("Loading ML artifacts…")

    try:
        model           = _load_artifact(MODEL_PATH)
        feature_columns = _load_artifact(FEATURE_PATH)
        threshold       = float(_load_artifact(THRESHOLD_PATH, default=0.5))
        imputer         = _load_artifact(IMPUTER_PATH, default=None)
        scaler          = _load_artifact(SCALER_PATH, default=None)
        logger.info("Model loaded (type=%s, features=%d, threshold=%.4f)",
                    type(model).__name__, len(feature_columns), threshold)
        logger.info(
            "Preprocessing loaded (imputer=%s, scaler=%s)",
            imputer is not None,
            scaler is not None,
        )
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc)
        model = None
        feature_columns = []

    # Build SHAP explainer at startup — auto-detect correct type for model
    if model is not None:
        try:
            t_shap = time.perf_counter()
            explainer = _build_explainer(model, feature_columns)
            logger.info(
                "SHAP %s ready (%.2fs)",
                type(explainer).__name__,
                time.perf_counter() - t_shap,
            )
        except Exception as exc:
            logger.warning("SHAP unavailable, falling back to KernelExplainer stub: %s", exc)
            explainer = None

    logger.info("Startup complete (%.2fs total)", time.perf_counter() - t0)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Prenava – Stunting Risk Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: per-request timing log
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("%s %s → %d  (%dms)",
                request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_shap_top_factors(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Return top-N SHAP factors with impact, value, and human message."""
    if explainer is None:
        return None
    try:
        import numpy as np
        t0 = time.perf_counter()
        raw = explainer.shap_values(df)

        # Normalize to a flat 1D array for the first (only) sample
        if isinstance(raw, list):
            sv = np.array(raw[1])
        else:
            sv = np.array(raw)

        if sv.ndim == 2:
            sv = sv[0]

        feature_names = df.columns.tolist()
        pairs = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_factors = []
        for feature, val in pairs[:top_n]:
            impact = "increase_risk" if val > 0 else "decrease_risk"
            input_val = float(df[feature].iloc[0])
            
            # Simple human message based on impact
            msg = f"{feature} berkontribusi terhadap {impact.replace('_', ' ')}"
            
            top_factors.append({
                "feature": feature,
                "impact": impact,
                "value": input_val,
                "message": msg
            })

        logger.info("SHAP computed (%.0fms)", (time.perf_counter() - t0) * 1000)
        return {
            "method": "SHAP",
            "top_factors": top_factors
        }
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


def _prepare_input_dataframe(payload: dict) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {', '.join(missing)}",
        )
    row = {col: float(payload[col]) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def _transform_for_model(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df
    if imputer is not None:
        transformed = imputer.transform(transformed)
    if scaler is not None:
        transformed = scaler.transform(transformed)
    if isinstance(transformed, np.ndarray):
        transformed = pd.DataFrame(
            transformed,
            columns=feature_columns,
            index=df.index,
        )
    return transformed


def _assert_ready():
    if model is None or not feature_columns:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Prenava Stunting Risk Prediction API", "version": "2.0.0"}


@app.get("/health")
def health():
    _assert_ready()
    return {
        "status": "healthy",
        "model_type": type(model).__name__,
        "features": len(feature_columns),
        "shap_ready": explainer is not None,
    }


@app.get("/warmup")
def warmup():
    """
    Endpoint to warm up the model with a dummy prediction.
    Call this after deploy to eliminate cold-start latency.
    """
    _assert_ready()
    dummy = pd.DataFrame([{col: 0.0 for col in feature_columns}], columns=feature_columns)
    model.predict_proba(_transform_for_model(dummy))
    return {"status": "warm"}


@app.get("/features")
def features():
    if not feature_columns:
        raise HTTPException(status_code=503, detail="Features not loaded.")
    return {"features": feature_columns, "count": len(feature_columns)}


@app.post("/predict")
def predict(data: StuntingInput):
    _assert_ready()
    t_total = time.perf_counter()

    try:
        # 1. Parse payload and align to trained feature contract
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        df_raw = _prepare_input_dataframe(payload)
        df_model = _transform_for_model(df_raw)

        # 2. Model inference
        t_inf = time.perf_counter()
        raw_proba = model.predict_proba(df_model)
        probability = float(raw_proba[0][1])
        prediction = int(probability >= threshold)
        inf_ms = round((time.perf_counter() - t_inf) * 1000)
        logger.info(
            "Inference done (prob=%.4f, threshold=%.2f, pred=%d, %dms)",
            probability,
            threshold,
            prediction,
            inf_ms,
        )

        # 3. SHAP explanation
        explanation = _compute_shap_top_factors(df_model, top_n=5) if explainer is not None else None
        total_ms = round((time.perf_counter() - t_total) * 1000)
        logger.info("Prediction request complete (%dms)", total_ms)

        return {
            "prediction": prediction,
            "risk_label": "high_risk" if prediction == 1 else "low_risk",
            "probability": round(probability, 4),
            "threshold": round(float(threshold), 4),
            "model_version": f"{type(model).__name__}_v1",
            "explanation": explanation,
            "recommendations": [],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Inference error."
        )