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

__all__ = [
    # MongoDB
    'MONGO_URI',
    'DB_NAME',

    # Spark
    'SPARK_JAR_PATH',
    'SPARK_DRIVER_MEMORY_DEFAULT',
    'SPARK_DRIVER_MEMORY_LARGE',

    # MLflow
    'MLFLOW_TRACKING_URI',

    # Collections
    'CYCLIC_INPUT_COLLECTION',
    'CYCLIC_OUTPUT_COLLECTION',
    'SPLIT_COLLECTION_PREFIX',
    'SPLIT_COLLECTION_SUFFIX_INPUT',
    'SPLIT_COLLECTION_SUFFIX_OUTPUT',

    # Helpers
    'get_split_collection_name',
    'get_mongo_config',
    'get_spark_config',
    'get_mlflow_config',
]
