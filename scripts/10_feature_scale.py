"""
Feature Standardization Selection Script (Stage 10)

Selects optimal EWMA half-life for feature standardization using CPCV splits.

Input: split_X_output collections with 18 features (after transformation)
Processes: 16 features (excludes volatility and fwd_logret_1)
Output: Half-life selections for 16 features

Exclusions from standardization:
- volatility: Keep original scale (meaningful interpretation)
- fwd_logret_1: Target variable, keep original scale

Usage:
    python scripts/10_feature_scale.py
"""

import os
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Apply Windows UTF-8 fix and MLflow patch
from src.utils.mlflow_patch import apply_mlflow_patch
apply_mlflow_patch()

import mlflow

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session
from src.feature_standardization import (
    EWMAHalfLifeProcessor,
    aggregate_across_splits,
    select_final_half_lives,
    identify_feature_names_from_collection
)
from src.feature_standardization.mlflow_logger import (
    log_split_results,
    log_aggregated_results
)

# Import centralized configuration
from src.config import (
    DB_NAME,
    MONGO_URI,
    SPARK_JAR_PATH,
    SPARK_DRIVER_MEMORY_DEFAULT,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENTS,
    SPLIT_COLLECTION_PREFIX,
    SPLIT_COLLECTION_SUFFIX_INPUT,
)

# =================================================================================================
# Configuration
# =================================================================================================

# Collection naming (from central config)
INPUT_COLLECTION_PREFIX = SPLIT_COLLECTION_PREFIX
INPUT_COLLECTION_SUFFIX = SPLIT_COLLECTION_SUFFIX_INPUT

# MLflow configuration (from central config)
MLFLOW_EXPERIMENT_NAME = MLFLOW_EXPERIMENTS['feature_standardization']

# Processing configuration
HALF_LIFE_CANDIDATES = [5, 10, 20, 40, 60]

# Spark configuration (from central config)
JAR_FILES_PATH = SPARK_JAR_PATH
DRIVER_MEMORY = SPARK_DRIVER_MEMORY_DEFAULT

# =================================================================================================
# Feature Filtering
# =================================================================================================

def filter_standardizable_features(feature_names):
    """
    Filter features to only include those that should be standardized.
    
    Excludes:
    - volatility: Keep original scale (meaningful interpretation)
    - fwd_logret_*: Target variable, keep original scale
    
    Args:
        feature_names: List of all feature names (18 features from transformation)
        
    Returns:
        List of features to standardize (16 features)
    """
    EXCLUDE_PATTERNS = [
        'fwd_logret_',      # Forward returns (targets)
    ]
    
    EXCLUDE_EXACT = []       # Keep original scale
    
    standardizable = []
    
    for feat_name in feature_names:
        # Skip exact matches
        if feat_name in EXCLUDE_EXACT:
            continue
        
        # Skip pattern matches
        if any(feat_name.startswith(pattern) for pattern in EXCLUDE_PATTERNS):
            continue
        
        standardizable.append(feat_name)
    
    excluded_count = len(feature_names) - len(standardizable)
    logger(f'Filtered to {len(standardizable)} standardizable features (excluded {excluded_count})', "INFO")
    
    if excluded_count > 0:
        logger(f'Excluded from standardization: volatility, fwd_logret_1', "INFO")
    
    return standardizable


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    log_section('FEATURE STANDARDIZATION SELECTION (STAGE 10)')
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark session
    logger('', "INFO")
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="FeatureStandardizationSelection",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Get feature names from first split using aggregation
        logger('', "INFO")
        logger('Identifying features...', "INFO")
        first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
        logger(f'Reading feature names from collection: {first_split}', "INFO")

        # Use aggregation-based function to avoid Spark schema inference issues
        all_feature_names = identify_feature_names_from_collection(
            spark=spark,
            db_name=DB_NAME,
            collection=first_split
        )
        logger(f'Total features in split collections: {len(all_feature_names)}', "INFO")
        
        # Filter to standardizable features only
        feature_names = filter_standardizable_features(all_feature_names)

        # Discover splits from MongoDB
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        all_collections = db.list_collection_names()

        split_ids = []
        for coll_name in all_collections:
            if coll_name.startswith("split_") and coll_name.endswith("_input"):
                split_id_str = coll_name.replace("split_", "").replace("_input", "")
                try:
                    split_id = int(split_id_str)
                    split_ids.append(split_id)
                except ValueError:
                    pass

        split_ids = sorted(split_ids)
        logger(f'Discovered {len(split_ids)} splits: {split_ids}', "INFO")
        client.close()

        if not split_ids:
            logger("ERROR: No split_X_input collections found!", "ERROR")
            return

        logger(f'Processing {len(feature_names)} standardizable features across {len(split_ids)} splits', "INFO")
        logger(f'Testing half-life values: {HALF_LIFE_CANDIDATES}', "INFO")

        # Initialize processor
        processor = EWMAHalfLifeProcessor(
            spark=spark,
            db_name=DB_NAME,
            input_collection_prefix=INPUT_COLLECTION_PREFIX,
            input_collection_suffix=INPUT_COLLECTION_SUFFIX
        )
        
        # Process each split
        all_split_results = {}

        for split_id in split_ids:
            # Process split - pass both full and standardizable feature lists
            split_results = processor.process_split(
                split_id=split_id,
                feature_names=all_feature_names,  # Full list for array validation
                standardizable_features=feature_names  # Standardizable features only
            )
            all_split_results[split_id] = split_results

            # Log to MLflow
            # Note: processor doesn't expose train_sample_rate, default is 0.1 (10%)
            log_split_results(split_id, split_results, train_sample_rate=0.1)
        
        # Aggregate results across splits
        logger('', "INFO")
        log_section('AGGREGATING RESULTS ACROSS SPLITS')
        aggregated = aggregate_across_splits(all_split_results)
        
        # Select best half-lives per feature
        logger('', "INFO")
        logger('Selecting optimal half-lives per feature...', "INFO")
        final_half_lives = select_final_half_lives(aggregated, strategy='most_frequent')

        # Log aggregated results
        log_aggregated_results(aggregated, final_half_lives)
        
        # Save results
        results_dir = Path(REPO_ROOT) / 'artifacts' / 'ewma_halflife_selection'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        results_file = results_dir / 'standardization_selection.json'
        with open(results_file, 'w') as f:
            json.dump({
                'final_half_lives': final_half_lives,
                'aggregated_metrics': {
                    feat: {
                        'selected_half_life': final_half_lives.get(feat, 20),
                        'most_frequent_half_life': agg['most_frequent_half_life'],
                        'stability': float(agg['stability']),
                        'n_splits': agg['n_splits'],
                        'avg_scores': {int(k): float(v) for k, v in agg['avg_scores'].items()},
                        'frequency_count': agg['frequency_count']
                    }
                    for feat, agg in aggregated.items()
                }
            }, f, indent=2)
        
        logger(f'Results saved to: {results_file}', "INFO")
        
        log_section('STANDARDIZATION SELECTION COMPLETED')
        logger(f'Selected half-lives for {len(final_half_lives)} features', "INFO")
        logger(f'Excluded features (keep original scale): volatility, fwd_logret_1', "INFO")
        
    except Exception as e:
        logger(f'Error during standardization selection: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()


if __name__ == "__main__":
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    main()