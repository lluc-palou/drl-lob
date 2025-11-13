# Pipeline Configuration

This module provides centralized configuration for the entire DRL-LOB pipeline.

## Overview

Previously, configuration values (MongoDB URIs, Spark paths, MLflow settings) were duplicated across 18+ scripts. This module centralizes all configuration to improve maintainability and consistency.

## Usage

### Basic Import

```python
from src.config import (
    MONGO_URI,
    DB_NAME,
    SPARK_JAR_PATH,
    MLFLOW_TRACKING_URI,
    get_spark_config,
)
```

### Get Pre-configured Settings

```python
# Get MongoDB configuration
mongo_config = get_mongo_config()
# Returns: {'mongo_uri': '...', 'db_name': '...'}

# Get Spark configuration
spark_config = get_spark_config('MyApp', '8g')
# Returns: {'app_name': 'MyApp', 'mongo_uri': '...', 'driver_memory': '8g', ...}

# Get MLflow configuration
mlflow_config = get_mlflow_config('feature_transformation')
# Returns: {'tracking_uri': '...', 'experiment_name': 'Feature_Transformation'}
```

### Collection Names

```python
from src.config import (
    CYCLIC_INPUT_COLLECTION,      # 'input'
    CYCLIC_OUTPUT_COLLECTION,      # 'output'
    SPLIT_COLLECTION_PREFIX,       # 'split_'
    SPLIT_COLLECTION_SUFFIX_INPUT, # '_input'
    get_split_collection_name,
)

# Generate split collection names
collection = get_split_collection_name(0, '_input')  # 'split_0_input'
```

## Environment Variables

You can override default configurations using environment variables:

- `MONGO_URI`: MongoDB connection string (default: `mongodb://127.0.0.1:27017/`)
- `MONGO_DB_NAME`: Database name (default: `raw`)
- `SPARK_JAR_PATH`: Path to Spark JAR files (default: `file:///C:/Users/llucp/spark_jars/`)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (default: `http://127.0.0.1:5000`)

Example:
```bash
export MONGO_URI="mongodb://myserver:27017/"
export SPARK_JAR_PATH="file:///opt/spark/jars/"
python scripts/06_materialize_splits.py
```

## Benefits

1. **Single Source of Truth**: All configuration in one place
2. **Easy Updates**: Change MongoDB URI once instead of 25+ times
3. **Environment-Specific**: Override via environment variables
4. **Type Safety**: Helper functions provide validated configurations
5. **Documentation**: Clear documentation of all settings

## Migration Guide

### Before (Old Pattern)
```python
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
```

### After (New Pattern)
```python
from src.config import (
    MONGO_URI,
    DB_NAME,
    MLFLOW_TRACKING_URI,
    SPARK_JAR_PATH as JAR_FILES_PATH,
)
```

Or use helper functions:
```python
from src.config import get_spark_config, SPARK_DRIVER_MEMORY_LARGE

SPARK_CONFIG = get_spark_config("MyApp", SPARK_DRIVER_MEMORY_LARGE)
```

## See Also

- `src/utils/mlflow_patch.py`: MLflow Windows UTF-8 fix utility
- `src/utils/spark.py`: Spark session creation utilities
- `src/utils/database.py`: Database utilities
