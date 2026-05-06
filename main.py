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
from pydantic import BaseModel, create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stunting_api")

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "stunting_model.pkl"
FEATURE_PATH    = BASE_DIR / "models" / "feature_columns.pkl"
THRESHOLD_PATH  = BASE_DIR / "models" / "stunting_threshold.pkl"

# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
model: object = None
feature_columns: list = []
threshold: float = 0.5
explainer: object = None
StuntingInput: type = None


def _load_artifact(path: Path, default=None):
    if path.exists():
        return joblib.load(path)
    if default is not None:
        return default
    raise FileNotFoundError(f"Missing artifact: {path}")


def _build_input_model(columns: list) -> type:
    if columns:
        return create_model(
            "StuntingInput",
            **{col: (float, ...) for col in columns},
        )

    class _Fallback(BaseModel):
        pass

    return _Fallback


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
    global model, feature_columns, threshold, explainer, StuntingInput

    t0 = time.perf_counter()
    logger.info("Loading ML artifacts…")

    try:
        model           = _load_artifact(MODEL_PATH)
        feature_columns = _load_artifact(FEATURE_PATH)
        threshold       = float(_load_artifact(THRESHOLD_PATH, default=0.5))
        logger.info("Model loaded (type=%s, features=%d, threshold=%.4f)",
                    type(model).__name__, len(feature_columns), threshold)
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

    StuntingInput = _build_input_model(feature_columns)
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
    """Return top-N SHAP factors as {feature: shap_value} (lightweight).

    Handles output from:
    - shap.LinearExplainer  → 2D ndarray (n_samples, n_features)
    - shap.TreeExplainer    → 2D ndarray OR list of 2 arrays for binary classifiers
    - shap.KernelExplainer  → 2D ndarray
    """
    if explainer is None:
        return {}
    try:
        import numpy as np
        t0 = time.perf_counter()
        raw = explainer.shap_values(df)

        # Normalize to a flat 1D array for the first (only) sample
        if isinstance(raw, list):
            # Binary classifier: list of [class0_vals, class1_vals]
            # Use class-1 (positive/stunting class) values
            sv = np.array(raw[1])
        else:
            sv = np.array(raw)

        # Shape may be (n_samples, n_features) or (n_features,)
        if sv.ndim == 2:
            sv = sv[0]  # first sample

        feature_names = df.columns.tolist()
        pairs = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        result = {k: round(v, 6) for k, v in pairs[:top_n]}
        logger.info("SHAP computed (%.0fms)", (time.perf_counter() - t0) * 1000)
        return result
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return {}



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
    dummy = pd.DataFrame([{col: 0.0 for col in feature_columns}])
    model.predict_proba(dummy)
    return {"status": "warm"}


@app.get("/features")
def features():
    if not feature_columns:
        raise HTTPException(status_code=503, detail="Features not loaded.")
    return {"features": feature_columns, "count": len(feature_columns)}


@app.post("/predict")
def predict(request: Request, data: StuntingInput):
    _assert_ready()

    t_total = time.perf_counter()

    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])[feature_columns]

        # --- Inference ---
        t_inf = time.perf_counter()
        probability = float(model.predict_proba(df)[0][1])
        prediction  = int(probability >= threshold)
        inf_ms = round((time.perf_counter() - t_inf) * 1000)

        # --- SHAP (top-5 only, lightweight) ---
        explanation = _compute_shap_top_factors(df, top_n=5)

        total_ms = round((time.perf_counter() - t_total) * 1000)
        logger.info(
            "predict OK  prob=%.4f pred=%d  inf=%dms  total=%dms",
            probability, prediction, inf_ms, total_ms,
        )

        return {
            "status":        "success",
            "prediction":    prediction,
            "risk_label":    "high_risk" if prediction == 1 else "low_risk",
            "probability":   round(probability, 4),
            "threshold":     threshold,
            "explanation":   explanation,
            "model_version": type(model).__name__,
            "latency_ms":    total_ms,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))