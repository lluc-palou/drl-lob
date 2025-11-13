"""
Apply EWMA Standardization Script

Applies EWMA standardization using selected half-life parameters from Stage 10.

This is Stage 11 in the pipeline - it follows half-life selection (Stage 10).

Usage:
    python scripts/11_apply_ewma_standardization.py
"""

import os
import sys
import json
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.feature_standardization import (
    identify_feature_names,
    filter_standardizable_features
)
from src.feature_standardization.apply_scaler import EWMAStandardizationApplicator

# Import per-split cyclic manager for collection cleanup
from src.split_materialization.per_split_cyclic_manager import PerSplitCyclicManager

# Import centralized configuration
from src.config import (
    DB_NAME,
    MONGO_URI,
    SPARK_JAR_PATH,
    SPARK_DRIVER_MEMORY_DEFAULT,
    SPLIT_COLLECTION_PREFIX,
    SPLIT_COLLECTION_SUFFIX_INPUT,
    SPLIT_COLLECTION_SUFFIX_OUTPUT,
    STANDARDIZATION_CONFIG,
)

# =================================================================================================
# Configuration
# =================================================================================================

# Collection naming (from central config)
INPUT_COLLECTION_PREFIX = SPLIT_COLLECTION_PREFIX
INPUT_COLLECTION_SUFFIX = SPLIT_COLLECTION_SUFFIX_INPUT  # Read from transformation output

OUTPUT_COLLECTION_PREFIX = SPLIT_COLLECTION_PREFIX
OUTPUT_COLLECTION_SUFFIX = SPLIT_COLLECTION_SUFFIX_OUTPUT  # Write for next stage (cyclic pattern)

# Path to half-life selection results (relative to repository root)
HALFLIFE_RESULTS_PATH = Path(REPO_ROOT) / "artifacts" / "ewma_halflife_selection" / "aggregation" / "final_halflifes.json"

# Standardization parameters (from central config)
CLIP_STD = STANDARDIZATION_CONFIG['clip_std']

# Spark configuration (from central config)
JAR_FILES_PATH = SPARK_JAR_PATH
DRIVER_MEMORY = SPARK_DRIVER_MEMORY_DEFAULT

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'

    logger('=' * 80, "INFO")
    logger('APPLY EWMA STANDARDIZATION', "INFO")
    logger('=' * 80, "INFO")
    
    # Load half-life results from Stage 10
    if not HALFLIFE_RESULTS_PATH.exists():
        logger(f'ERROR: Half-life results not found at {HALFLIFE_RESULTS_PATH}', "ERROR")
        logger('Please run Stage 10 (10_select_ewma_halflife.py) first', "ERROR")
        return
    
    with open(HALFLIFE_RESULTS_PATH, 'r') as f:
        final_halflifes = json.load(f)
    
    logger(f'Loaded half-lives for {len(final_halflifes)} features', "INFO")

    # Show sample half-lives
    sample_halflifes = list(final_halflifes.items())[:5]
    for feat, hl in sample_halflifes:
        logger(f'  {feat}: half_life={hl}', "INFO")
    if len(final_halflifes) > 5:
        logger(f'  ... and {len(final_halflifes) - 5} more', "INFO")

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

    if not split_ids:
        logger("ERROR: No split_X_input collections found!", "ERROR")
        client.close()
        return

    client.close()

    # Create Spark session
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="EWMAStandardization",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Get feature names from first split
        logger('Identifying features...', "INFO")
        first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
        logger(f'Loading from collection: {first_split}', "INFO")
        
        sample_df = (
            spark.read.format("mongodb")
            .option("database", DB_NAME)
            .option("collection", first_split)
            .load()
            .limit(1)
        )
        
        count = sample_df.count()
        if count == 0:
            raise ValueError(f"Collection '{first_split}' is empty!")
        
        all_feature_names = identify_feature_names(sample_df)
        
        logger(f'Total features: {len(all_feature_names)}', "INFO")
        logger(f'Standardizing: {len(final_halflifes)} features', "INFO")

        # Initialize applicator
        applicator = EWMAStandardizationApplicator(
            spark=spark,
            db_name=DB_NAME,
            final_halflifes=final_halflifes,
            clip_std=CLIP_STD
        )

        # Initialize collection manager for cleanup
        manager = PerSplitCyclicManager(MONGO_URI, DB_NAME)

        # Process each split
        for split_id in split_ids:
            logger('', "INFO")  # Blank line

            # Clear output collection before processing (prevent duplication)
            logger(f'Preparing split {split_id} for processing...', "INFO")
            manager.prepare_split_for_processing(split_id, force=True)

            # Apply standardization
            total_processed = applicator.apply_to_split(
                split_id=split_id,
                feature_names=all_feature_names,
                input_collection_prefix=INPUT_COLLECTION_PREFIX,
                input_collection_suffix=INPUT_COLLECTION_SUFFIX,
                output_collection_prefix=OUTPUT_COLLECTION_PREFIX,
                output_collection_suffix=OUTPUT_COLLECTION_SUFFIX
            )
            
            # Reset scalers for next split (each split processes independently)
            applicator.scalers = {
                feat: type(scaler)(scaler.half_life)
                for feat, scaler in applicator.scalers.items()
            }
        
        # Summary
        logger('', "INFO")
        logger('=' * 80, "INFO")
        logger('EWMA STANDARDIZATION COMPLETE', "INFO")
        logger('=' * 80, "INFO")
        logger(f'Processed {len(split_ids)} splits', "INFO")

        # ✅ FIX: Use consistent swap logic from cyclic manager
        logger('', "INFO")
        logger('Swapping collections for cyclic pattern...', "INFO")

        for split_id in split_ids:
            manager.swap_split_to_input(split_id)

        manager.close()

        logger('', "INFO")
        logger('Collection swapping complete', "INFO")
        logger(f'Next stage will read from: {INPUT_COLLECTION_PREFIX}{{split_ids}}{INPUT_COLLECTION_SUFFIX}', "INFO")
        
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    main()