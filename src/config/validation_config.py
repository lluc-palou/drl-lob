"""
Validation and Cross-Validation Configuration

Central configuration for CPCV, temporal parameters, and stylized facts testing.
Used by stages 03 (data splitting) and 09 (stylized facts testing).
"""

import os

# =================================================================================================
# CPCV (Combinatorial Purged Cross-Validation) Parameters
# =================================================================================================

# Master seed for reproducibility
MASTER_SEED = int(os.environ.get('CPCV_MASTER_SEED', 42))

# Train/test split ratio
TEST_RATIO = float(os.environ.get('CPCV_TEST_RATIO', 0.20))

# Number of temporal folds for CPCV
N_FOLDS = int(os.environ.get('CPCV_N_FOLDS', 10))

# Number of validation folds (k in CPCV k-fold validation)
K_VALIDATION_FOLDS = int(os.environ.get('CPCV_K_VALIDATION_FOLDS', 2))

# Complete CPCV configuration dict
CPCV_CONFIG = {
    'master_seed': MASTER_SEED,
    'test_ratio': TEST_RATIO,
    'n_folds': N_FOLDS,
    'k_validation_folds': K_VALIDATION_FOLDS,
}

# =================================================================================================
# Temporal Parameters
# =================================================================================================

# Sampling interval in seconds (data aggregation window)
SAMPLING_INTERVAL_SECONDS = int(os.environ.get('SAMPLING_INTERVAL_SECONDS', 30))

# Context length in samples (number of past samples for prediction)
CONTEXT_LENGTH_SAMPLES = int(os.environ.get('CONTEXT_LENGTH_SAMPLES', 240))

# Forecast horizon in steps (how far ahead to predict)
FORECAST_HORIZON_STEPS = int(os.environ.get('FORECAST_HORIZON_STEPS', 240))

# Purge length in samples (gap between train and test to prevent leakage)
PURGE_LENGTH_SAMPLES = int(os.environ.get('PURGE_LENGTH_SAMPLES', 240))

# Embargo length in samples (additional buffer after test period)
EMBARGO_LENGTH_SAMPLES = int(os.environ.get('EMBARGO_LENGTH_SAMPLES', 240))

# Complete temporal configuration dict
TEMPORAL_CONFIG = {
    'sampling_interval_seconds': SAMPLING_INTERVAL_SECONDS,
    'context_length_samples': CONTEXT_LENGTH_SAMPLES,
    'forecast_horizon_steps': FORECAST_HORIZON_STEPS,
    'purge_length_samples': PURGE_LENGTH_SAMPLES,
    'embargo_length_samples': EMBARGO_LENGTH_SAMPLES,
}

# =================================================================================================
# Stylized Facts Testing Parameters
# =================================================================================================

# Window length for representative windows extraction (in samples)
STYLIZED_FACTS_WINDOW_LENGTH = int(os.environ.get('STYLIZED_FACTS_WINDOW_LENGTH', 60))

# Edge margin to avoid boundary effects (in samples)
STYLIZED_FACTS_EDGE_MARGIN = int(os.environ.get('STYLIZED_FACTS_EDGE_MARGIN', 25))

# Statistical significance level for hypothesis tests
SIGNIFICANCE_LEVEL = float(os.environ.get('SIGNIFICANCE_LEVEL', 0.05))

# Confidence level for confidence intervals
CONFIDENCE_LEVEL = float(os.environ.get('CONFIDENCE_LEVEL', 0.95))

# Enable enhanced statistical analysis
ENABLE_ENHANCED_STATS = os.environ.get('ENABLE_ENHANCED_STATS', 'true').lower() == 'true'

# Complete stylized facts configuration dict
STYLIZED_FACTS_CONFIG = {
    'window_length_samples': STYLIZED_FACTS_WINDOW_LENGTH,
    'edge_margin_samples': STYLIZED_FACTS_EDGE_MARGIN,
    'forecast_horizon': FORECAST_HORIZON_STEPS,  # Reuse from temporal config
    'significance_level': SIGNIFICANCE_LEVEL,
    'confidence_level': CONFIDENCE_LEVEL,
    'enable_enhanced_stats': ENABLE_ENHANCED_STATS,
}

# =================================================================================================
# Complete Stage 03 Configuration (for backwards compatibility)
# =================================================================================================

def get_data_splitting_config():
    """
    Get complete configuration for Stage 03 (Data Splitting).

    Returns:
        Dictionary with all CPCV and temporal parameters
    """
    return {
        'experiment_id': None,  # Set by MLflow if tracking
        'master_seed': MASTER_SEED,
        'temporal_params': TEMPORAL_CONFIG,
        'train_test_split': {'test_ratio': TEST_RATIO},
        'cpcv': {
            'n_folds': N_FOLDS,
            'k_validation_folds': K_VALIDATION_FOLDS,
        },
        'stylized_facts': {
            'window_length_samples': STYLIZED_FACTS_WINDOW_LENGTH,
            'edge_margin_samples': STYLIZED_FACTS_EDGE_MARGIN,
        },
    }
