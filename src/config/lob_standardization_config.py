"""
LOB Standardization Configuration

Central configuration for LOB quantization and standardization parameters.
Used by stage 05 (LOB standardization).
"""

import os

# =================================================================================================
# Quantization Parameters
# =================================================================================================

# Number of bins for price/volume quantization
# Output will have B+1 bins (0 to B inclusive)
B = int(os.environ.get('LOB_QUANTIZATION_BINS', 1000))

# Price clipping threshold (in standard deviations)
# Prices beyond ±delta*std will be clipped
DELTA = float(os.environ.get('LOB_PRICE_CLIP_THRESHOLD', 1000.0))

# Minimum price spacing near zero (prevents division by zero)
EPSILON = float(os.environ.get('LOB_MIN_PRICE_SPACING', 1.0))

# =================================================================================================
# Standardization Safety Parameters
# =================================================================================================

# Small constant to avoid division by zero
EPS = float(os.environ.get('LOB_EPSILON', 1e-8))

# Minimum denominator value for normalization
MIN_DENOM = float(os.environ.get('LOB_MIN_DENOMINATOR', 1e-6))

# =================================================================================================
# Batch Processing Parameters
# =================================================================================================

# Required past hours for volatility calculation
REQUIRED_PAST_HOURS = int(os.environ.get('LOB_REQUIRED_PAST_HOURS', 2))

# =================================================================================================
# Analysis Mode
# =================================================================================================

# Enable volume coverage analysis (for debugging/analysis only)
VOLUME_COVERAGE_ANALYSIS = os.environ.get('VOLUME_COVERAGE_ANALYSIS', 'false').lower() == 'true'

# =================================================================================================
# Complete Configuration
# =================================================================================================

# Complete LOB standardization configuration
LOB_STANDARDIZATION_CONFIG = {
    # Quantization parameters
    'B': B,
    'delta': DELTA,
    'epsilon': EPSILON,

    # Safety parameters
    'eps': EPS,
    'min_denom': MIN_DENOM,

    # Batch processing
    'required_past_hours': REQUIRED_PAST_HOURS,

    # Analysis mode
    'volume_coverage_analysis': VOLUME_COVERAGE_ANALYSIS,
}

# Additional Spark configurations for LOB standardization
LOB_STANDARDIZATION_SPARK_CONFIGS = {
    "spark.network.timeout": "300s",
    "spark.executor.heartbeatInterval": "60s",
    "spark.mongodb.connection.timeout.ms": "30000",
    "spark.mongodb.socket.timeout.ms": "120000",
    "spark.mongodb.write.retryWrites": "true",
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def get_lob_standardization_config():
    """
    Get complete LOB standardization configuration.

    Returns:
        Dictionary with all LOB standardization parameters
    """
    return LOB_STANDARDIZATION_CONFIG.copy()


def get_quantization_params():
    """
    Get quantization parameters only.

    Returns:
        Dictionary with B, delta, epsilon
    """
    return {
        'B': B,
        'delta': DELTA,
        'epsilon': EPSILON,
    }


def get_output_bins_count():
    """
    Get the number of output bins after quantization.

    Returns:
        Number of bins (B+1)
    """
    return B + 1
