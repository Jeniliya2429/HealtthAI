from .risk_assessment import (
    classify_risk, apply_cross_disease_amplifier,
    compute_health_score, simulate_risk_progression,
    get_health_tips, get_diet_recommendations, get_age_group
)
from .preprocessing import (
    load_model, predict_disease, get_feature_importance,
    load_metrics, models_exist, partial_models_exist
)
