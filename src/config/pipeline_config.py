"""
Pipeline Configuration Module

Central configuration for all pipeline settings including MongoDB, Spark, and MLflow.
All hardcoded values should be defined here and imported by scripts.

Environment Variables (optional overrides):
    MONGO_URI: MongoDB connection string
    MONGO_DB_NAME: Database name
    SPARK_JAR_PATH: Path to Spark JAR files
    MLFLOW_TRACKING_URI: MLflow tracking server URI
"""

import os

# =================================================================================================
# MongoDB Configuration
# =================================================================================================

# MongoDB connection URI
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://127.0.0.1:27017/')

# Database name
DB_NAME = os.environ.get('MONGO_DB_NAME', 'raw')

# =================================================================================================
# Spark Configuration
# =================================================================================================

# Path to Spark MongoDB connector JARs
# Note: This should be updated for your environment or use an environment variable
SPARK_JAR_PATH = os.environ.get(
    'SPARK_JAR_PATH',
    'file:///C:/Users/llucp/spark_jars/'  # Default Windows path
)

# Spark driver memory configurations
SPARK_DRIVER_MEMORY_DEFAULT = '4g'  # For lighter operations
SPARK_DRIVER_MEMORY_LARGE = '8g'    # For heavy operations (materialization, VQ-VAE, etc.)

# =================================================================================================
# MLflow Configuration
# =================================================================================================

# MLflow tracking server URI
MLFLOW_TRACKING_URI = os.environ.get(
    'MLFLOW_TRACKING_URI',
    'http://127.0.0.1:5000'  # Default to remote tracking server
)

# MLflow experiment names (by pipeline stage)
MLFLOW_EXPERIMENTS = {
    'feature_transformation': 'Feature_Transformation',
    'feature_standardization': 'Feature_Standardization',
    'vqvae_hyperparameter_search': 'VQ-VAE_Hyperparameter_Search',
    'vqvae_production': 'VQ-VAE_Production_Training',
    'prior_production': 'LOB_Prior_Production_Training',
    'synthetic_generation': 'LOB_Synthetic_Generation',
}

# =================================================================================================
# Collection Naming Patterns
# =================================================================================================

# Cyclic pipeline collections (Stages 2-5)
CYCLIC_INPUT_COLLECTION = 'input'
CYCLIC_OUTPUT_COLLECTION = 'output'

# Split collection patterns (Stages 6+)
SPLIT_COLLECTION_PREFIX = 'split_'
SPLIT_COLLECTION_SUFFIX_INPUT = '_input'
SPLIT_COLLECTION_SUFFIX_OUTPUT = '_output'

# Special collections
TEST_DATA_COLLECTION = 'test_data'

# =================================================================================================
# Helper Functions
# =================================================================================================

def get_split_collection_name(split_id: int, suffix: str = '_input') -> str:
    """
    Generate split collection name.

    Args:
        split_id: Split ID number
        suffix: Collection suffix ('_input' or '_output')

    Returns:
        Collection name (e.g., 'split_0_input')
    """
    return f"{SPLIT_COLLECTION_PREFIX}{split_id}{suffix}"


def get_mongo_config() -> dict:
    """
    Get MongoDB configuration dictionary.

    Returns:
        Dictionary with mongo_uri and db_name
    """
    return {
        'mongo_uri': MONGO_URI,
        'db_name': DB_NAME,
    }


def get_spark_config(app_name: str, driver_memory: str = SPARK_DRIVER_MEMORY_DEFAULT) -> dict:
    """
    Get Spark configuration dictionary.

    Args:
        app_name: Spark application name
        driver_memory: Driver memory size (e.g., '4g', '8g')

    Returns:
        Dictionary with Spark configuration
    """
    return {
        'app_name': app_name,
        'mongo_uri': MONGO_URI,
        'db_name': DB_NAME,
        'driver_memory': driver_memory,
        'jar_files_path': SPARK_JAR_PATH,
    }


def get_mlflow_config(experiment_key: str) -> dict:
    """
    Get MLflow configuration dictionary.

    Args:
        experiment_key: Key from MLFLOW_EXPERIMENTS dict

    Returns:
        Dictionary with MLflow configuration
    """
    return {
        'tracking_uri': MLFLOW_TRACKING_URI,
        'experiment_name': MLFLOW_EXPERIMENTS.get(
            experiment_key,
            f'Pipeline_{experiment_key}'  # Fallback
        ),
    }


# =================================================================================================
# Pipeline Stage Configuration
# =================================================================================================

# Stage-specific Spark memory requirements
STAGE_MEMORY_CONFIG = {
    2: SPARK_DRIVER_MEMORY_DEFAULT,  # Data Ingestion
    3: SPARK_DRIVER_MEMORY_DEFAULT,  # Data Splitting
    4: SPARK_DRIVER_MEMORY_LARGE,    # Feature Derivation
    5: SPARK_DRIVER_MEMORY_LARGE,    # LOB Standardization
    6: SPARK_DRIVER_MEMORY_LARGE,    # Materialize Splits
    7: SPARK_DRIVER_MEMORY_DEFAULT,  # Feature Transform Selection
    8: SPARK_DRIVER_MEMORY_DEFAULT,  # Apply Transforms
    9: SPARK_DRIVER_MEMORY_DEFAULT,  # Stylized Facts
    10: SPARK_DRIVER_MEMORY_DEFAULT, # EWMA Selection
    11: SPARK_DRIVER_MEMORY_DEFAULT, # Apply EWMA
    12: SPARK_DRIVER_MEMORY_DEFAULT, # Filter Nulls
    13: SPARK_DRIVER_MEMORY_LARGE,   # VQ-VAE Hyperparameter Search
    14: SPARK_DRIVER_MEMORY_LARGE,   # VQ-VAE Production
    15: SPARK_DRIVER_MEMORY_LARGE,   # Prior Production
    16: SPARK_DRIVER_MEMORY_LARGE,   # Synthetic Generation
}


def get_stage_spark_config(stage_num: int, app_name: str = None) -> dict:
    """
    Get Spark configuration for a specific pipeline stage.

    Args:
        stage_num: Pipeline stage number (2-16)
        app_name: Optional app name override

    Returns:
        Dictionary with Spark configuration optimized for the stage
    """
    if app_name is None:
        app_name = f"Stage_{stage_num}"

    memory = STAGE_MEMORY_CONFIG.get(stage_num, SPARK_DRIVER_MEMORY_DEFAULT)
    return get_spark_config(app_name, memory)
