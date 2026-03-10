"""
=============================================================================
  AI Multi-Disease Risk Assessment System
  FILE: utils/preprocessing.py  |  v2.0
  Handles: Safe model loading, safe prediction, feature importance
=============================================================================
"""
import numpy as np
import pickle
import os
import json

MODELS_DIR = "models"


def load_model(disease_key: str):
    """Safely load model pipeline and feature columns."""
    path = os.path.join(MODELS_DIR, f"{disease_key}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {disease_key}. Please run train_models.py first.")
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['pipeline'], obj['feature_cols']


def predict_disease(disease_key: str, input_values: list) -> dict:
    """
    Safely run prediction and return probability.
    Returns a dict with probability and status.
    Never raises to the caller — all errors returned in 'error' key.
    """
    try:
        pipeline, feature_cols = load_model(disease_key)
        if len(input_values) != len(feature_cols):
            return {
                "success": False,
                "error": f"Input mismatch: expected {len(feature_cols)} values, got {len(input_values)}",
                "probability": 0.5,
            }
        X = np.array(input_values, dtype=float).reshape(1, -1)
        # Replace any NaN/Inf with column medians (safe fallback)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        prob_array = pipeline.predict_proba(X)[0]
        prob_positive = float(prob_array[1]) if len(prob_array) > 1 else float(prob_array[0])
        prob_positive = float(np.clip(prob_positive, 0.0, 1.0))
        return {
            "success"     : True,
            "probability" : prob_positive,
            "all_probs"   : prob_array.tolist(),
            "feature_cols": feature_cols,
            "error"       : None,
        }
    except FileNotFoundError as e:
        return {"success": False, "error": str(e), "probability": 0.5}
    except ValueError as e:
        return {"success": False, "error": f"Value error: {str(e)}", "probability": 0.5}
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}", "probability": 0.5}


def get_feature_importance(disease_key: str) -> dict:
    """Safely extract feature importances from trained RandomForest."""
    try:
        pipeline, feature_cols = load_model(disease_key)
        clf = pipeline.named_steps.get('classifier')
        if clf is not None and hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            sorted_idx  = np.argsort(importances)[::-1]
            return {
                feature_cols[i]: round(float(importances[i]) * 100, 2)
                for i in sorted_idx[:8]
            }
        return {}
    except Exception:
        return {}


def load_metrics(disease_key: str = None) -> dict:
    """Safely load evaluation metrics. Returns empty dict on failure."""
    try:
        if disease_key:
            path = os.path.join(MODELS_DIR, f"{disease_key}_meta.json")
        else:
            path = os.path.join(MODELS_DIR, 'all_metrics.json')
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def models_exist() -> bool:
    """Check if ALL required model files are present."""
    required = ['diabetes','heart','parkinsons','ckd','liver',
                'stroke','breast_cancer','thyroid','pcos','anemia']
    return all(
        os.path.exists(os.path.join(MODELS_DIR, f"{k}_model.pkl"))
        for k in required
    )


def partial_models_exist() -> list:
    """Return list of disease keys whose models exist."""
    all_keys = ['diabetes','heart','parkinsons','ckd','liver',
                'stroke','breast_cancer','thyroid','pcos','anemia']
    return [k for k in all_keys
            if os.path.exists(os.path.join(MODELS_DIR, f"{k}_model.pkl"))]
