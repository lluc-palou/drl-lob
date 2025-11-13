"""
Feature Engineering Configuration

Central configuration for feature derivation, transformation, and standardization.
Used by stages 04, 06, 07, 08, 10, 11.
"""

import os

# =================================================================================================
# Feature Derivation Parameters (Stage 04)
# =================================================================================================

# Forward return horizons (in steps)
FORWARD_HORIZONS = [1]  # Immediate 1-step ahead only

# Historical lag periods for returns (in steps)
HISTORICAL_LAGS = [1, 2, 3, 5, 10, 20]  # Cut at 20 (white noise after)

# Half-life for variance calculation (EWMA)
VARIANCE_HALF_LIFE = int(os.environ.get('VARIANCE_HALF_LIFE', 20))

# Depth aggregation bands (top N levels)
DEPTH_BANDS = [5, 50]  # top_5 and top_50 only

# Decision lag (execution delay in steps)
DECISION_LAG = int(os.environ.get('DECISION_LAG', 0))

# Required past hours for feature calculation
REQUIRED_PAST_HOURS = int(os.environ.get('REQUIRED_PAST_HOURS', 3))

# Required future hours for forward-looking features
REQUIRED_FUTURE_HOURS = int(os.environ.get('REQUIRED_FUTURE_HOURS', 1))

# Complete feature derivation configuration
FEATURE_DERIVATION_CONFIG = {
    'forward_horizons': FORWARD_HORIZONS,
    'historical_lags': HISTORICAL_LAGS,
    'variance_half_life': VARIANCE_HALF_LIFE,
    'depth_bands': DEPTH_BANDS,
    'decision_lag': DECISION_LAG,
    'required_past_hours': REQUIRED_PAST_HOURS,
    'required_future_hours': REQUIRED_FUTURE_HOURS,
}

# Additional Spark configurations for feature derivation
FEATURE_DERIVATION_SPARK_CONFIGS = {
    "spark.network.timeout": "300s",
    "spark.executor.heartbeatInterval": "60s",
    "spark.mongodb.connection.timeout.ms": "30000",
    "spark.mongodb.socket.timeout.ms": "120000",
    "spark.mongodb.write.retryWrites": "true"
}

# =================================================================================================
# Feature Projection Parameters (Stage 06)
# =================================================================================================

# Features to EXCLUDE from projection (intermediate calculations not needed downstream)
EXCLUDE_FROM_PROJECTION = [
    'mid_price',        # Used to calculate log_return, then not needed
    'log_return',       # Used to calculate hist_logret_*, then not needed
    'variance_proxy',   # Used to calculate volatility, then not needed
    'spread',           # Nearly constant, not informative
]

# Features to EXCLUDE from transformation (keep original scale)
# These are kept in features array but not transformed in stages 07-08
EXCLUDE_FROM_TRANSFORMATION_PATTERNS = ['fwd_logret_']  # Target variable patterns
EXCLUDE_FROM_TRANSFORMATION_EXACT = []  # Exact feature names (e.g., 'volatility' if needed)

# Features to EXCLUDE from standardization (keep original scale)
# These are kept in features array but not standardized in stages 10-11
EXCLUDE_FROM_STANDARDIZATION_PATTERNS = ['fwd_logret_']  # Target variable patterns
EXCLUDE_FROM_STANDARDIZATION_EXACT = []  # Exact feature names

# Complete feature projection configuration
FEATURE_PROJECTION_CONFIG = {
    'exclude_from_projection': EXCLUDE_FROM_PROJECTION,
    'exclude_from_transformation_patterns': EXCLUDE_FROM_TRANSFORMATION_PATTERNS,
    'exclude_from_transformation_exact': EXCLUDE_FROM_TRANSFORMATION_EXACT,
    'exclude_from_standardization_patterns': EXCLUDE_FROM_STANDARDIZATION_PATTERNS,
    'exclude_from_standardization_exact': EXCLUDE_FROM_STANDARDIZATION_EXACT,
}

# =================================================================================================
# Feature Transformation Parameters (Stage 07-08)
# =================================================================================================

# Training sample rate for transformation selection (1.0 = use all training data)
TRAIN_SAMPLE_RATE = float(os.environ.get('TRAIN_SAMPLE_RATE', 1.0))

# Complete transformation configuration
TRANSFORMATION_CONFIG = {
    'train_sample_rate': TRAIN_SAMPLE_RATE,
}

# =================================================================================================
# Feature Standardization Parameters (Stage 10-11)
# =================================================================================================

# EWMA half-life candidates to test
HALF_LIFE_CANDIDATES = [5, 10, 20, 40, 60]

# Standard deviation clipping threshold (clip at ±N std deviations)
CLIP_STD = float(os.environ.get('CLIP_STD', 3.0))

# Complete standardization configuration
STANDARDIZATION_CONFIG = {
    'half_life_candidates': HALF_LIFE_CANDIDATES,
    'clip_std': CLIP_STD,
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def get_transformable_features(feature_names: list) -> list:
    """
    Filter features to only those that should be transformed.

    Args:
        feature_names: List of all feature names

    Returns:
        List of feature names to transform
    """
    transformable = []
    for feat in feature_names:
        # Check pattern exclusions
        if any(pattern in feat for pattern in EXCLUDE_FROM_TRANSFORMATION_PATTERNS):
            continue
        # Check exact exclusions
        if feat in EXCLUDE_FROM_TRANSFORMATION_EXACT:
            continue
        transformable.append(feat)
    return transformable


def get_standardizable_features(feature_names: list) -> list:
    """
    Filter features to only those that should be standardized.

    Args:
        feature_names: List of all feature names

    Returns:
        List of feature names to standardize
    """
    standardizable = []
    for feat in feature_names:
        # Check pattern exclusions
        if any(pattern in feat for pattern in EXCLUDE_FROM_STANDARDIZATION_PATTERNS):
            continue
        # Check exact exclusions
        if feat in EXCLUDE_FROM_STANDARDIZATION_EXACT:
            continue
        standardizable.append(feat)
    return standardizable


def get_projected_features(feature_names: list) -> list:
    """
    Filter features for projection to split collections.

    Args:
        feature_names: List of all feature names

    Returns:
        List of features to keep in projection
    """
    return [feat for feat in feature_names if feat not in EXCLUDE_FROM_PROJECTION]
