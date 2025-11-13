import os
import sys
import time

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Import utilities
from src.utils.spark import create_spark_session
from src.utils.logging import logger, log_section

# Import standardization classes
from src.lob_standardization import StandardizationOrchestrator

# Import centralized configuration
from src.config import (
    MONGO_URI,
    DB_NAME,
    CYCLIC_INPUT_COLLECTION,
    CYCLIC_OUTPUT_COLLECTION,
    LOB_STANDARDIZATION_CONFIG,
)
from src.config.lob_standardization_config import LOB_STANDARDIZATION_SPARK_CONFIGS

# =================================================================================================
# Configuration
# =================================================================================================

# Collection names (from central config)
INPUT_COLLECTION = CYCLIC_INPUT_COLLECTION  # Input from feature engineering stage
OUTPUT_COLLECTION = CYCLIC_OUTPUT_COLLECTION

# LOB standardization configuration (from central config)
CONFIG = LOB_STANDARDIZATION_CONFIG

# Additional Spark configurations (from central config)
ADDITIONAL_SPARK_CONFIGS = LOB_STANDARDIZATION_SPARK_CONFIGS

# Create Spark session
spark = create_spark_session(
    app_name="Stage5_LOB_Standardization",
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    driver_memory="8g",
    additional_configs=ADDITIONAL_SPARK_CONFIGS
)

# =================================================================================================
# Pipeline
# =================================================================================================

def run_standardization_pipeline():
    """
    Executes LOB standardization pipeline.
    """
    log_section("LOB Standardization Pipeline")
    logger(f'Input Collection: {INPUT_COLLECTION}', "INFO")
    logger(f'Output Collection: {OUTPUT_COLLECTION}', "INFO")
    log_section("", char="-")
    logger(f'Number of bins (B): {CONFIG["B"]} (output: {CONFIG["B"]+1} bins)', "INFO")
    logger(f'Price clipping threshold (delta): {CONFIG["delta"]} std', "INFO")
    logger(f'Minimum price spacing (epsilon): {CONFIG["epsilon"]}', "INFO")
    logger(f'Volume coverage analysis mode: {CONFIG["volume_coverage_analysis"]}', "INFO")
    log_section("", char="=")
    
    # Initialize orchestrator
    orchestrator = StandardizationOrchestrator(
        spark=spark,
        db_name=DB_NAME,
        input_collection=INPUT_COLLECTION,
        output_collection=OUTPUT_COLLECTION,
        config=CONFIG
    )
    
    # Load raw LOB data
    log_section("Stage 1: Loading Split LOB Data")
    orchestrator.load_raw_data()
    
    # Determine processable hours
    log_section("Stage 2: Determining Processable Hours")
    orchestrator.determine_processable_hours()
    
    # Process hourly batches
    log_section("Stage 3: Processing Hourly Batches")
    orchestrator.process_all_batches()
    
    log_section("Pipeline Complete")

# =================================================================================================
# Main Entry Point
# =================================================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # Checks if runned from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'

    try:
        run_standardization_pipeline()
        
        total_time = time.time() - start_time
        logger(f'Total execution time: {total_time:.2f} seconds', "INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()