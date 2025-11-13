"""
Configuration Module

Centralized configuration for the entire pipeline.
"""

from .pipeline_config import (
    # MongoDB settings
    MONGO_URI,
    DB_NAME,

    # Spark settings
    SPARK_JAR_PATH,
    SPARK_DRIVER_MEMORY_DEFAULT,
    SPARK_DRIVER_MEMORY_LARGE,

    # MLflow settings
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENTS,

    # Collection naming patterns
    CYCLIC_INPUT_COLLECTION,
    CYCLIC_OUTPUT_COLLECTION,
    SPLIT_COLLECTION_PREFIX,
    SPLIT_COLLECTION_SUFFIX_INPUT,
    SPLIT_COLLECTION_SUFFIX_OUTPUT,

    # Helper functions
    get_split_collection_name,
    get_mongo_config,
    get_spark_config,
    get_mlflow_config,
)

# Import validation configuration
from .validation_config import (
    CPCV_CONFIG,
    TEMPORAL_CONFIG,
    STYLIZED_FACTS_CONFIG,
    get_data_splitting_config,
)

# Import feature engineering configuration
from .feature_engineering_config import (
    FEATURE_DERIVATION_CONFIG,
    FEATURE_PROJECTION_CONFIG,
    TRANSFORMATION_CONFIG,
    STANDARDIZATION_CONFIG,
    get_transformable_features,
    get_standardizable_features,
    get_projected_features,
)

# Import LOB standardization configuration
from .lob_standardization_config import (
    LOB_STANDARDIZATION_CONFIG,
    get_lob_standardization_config,
)

__all__ = [
    # Pipeline config
    'MONGO_URI',
    'DB_NAME',
    'SPARK_JAR_PATH',
    'SPARK_DRIVER_MEMORY_DEFAULT',
    'SPARK_DRIVER_MEMORY_LARGE',
    'MLFLOW_TRACKING_URI',
    'MLFLOW_EXPERIMENTS',
    'CYCLIC_INPUT_COLLECTION',
    'CYCLIC_OUTPUT_COLLECTION',
    'SPLIT_COLLECTION_PREFIX',
    'SPLIT_COLLECTION_SUFFIX_INPUT',
    'SPLIT_COLLECTION_SUFFIX_OUTPUT',
    'get_split_collection_name',
    'get_mongo_config',
    'get_spark_config',
    'get_mlflow_config',

    # Validation config
    'CPCV_CONFIG',
    'TEMPORAL_CONFIG',
    'STYLIZED_FACTS_CONFIG',
    'get_data_splitting_config',

    # Feature engineering config
    'FEATURE_DERIVATION_CONFIG',
    'FEATURE_PROJECTION_CONFIG',
    'TRANSFORMATION_CONFIG',
    'STANDARDIZATION_CONFIG',
    'get_transformable_features',
    'get_standardizable_features',
    'get_projected_features',

    # LOB standardization config
    'LOB_STANDARDIZATION_CONFIG',
    'get_lob_standardization_config',
]
