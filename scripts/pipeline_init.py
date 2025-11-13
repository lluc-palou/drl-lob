"""
DEPRECATED: This script is no longer used.

The pipeline now starts from Stage 2 (data ingestion from parquet files),
not from a raw_lob archive collection.

To initialize the pipeline:
1. Run Stage 2: python scripts/02_data_ingestion.py
2. This will load data from parquet files and create the 'output' collection
3. Stage 2 will automatically swap output -> input
4. Continue with Stage 3 and onwards via run_pipeline.py

This script and the initialize_pipeline() method have been removed as they
relied on the obsolete raw_lob archive collection pattern.
"""

import os
import sys
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger, log_section

# =================================================================================================
# Configuration - Edit these settings
# =================================================================================================

CONFIG = {
    'mongo_uri': "mongodb://127.0.0.1:27017/",
    'db_name': "raw",
    'force': True,  # Set to True to drop existing working collections
}

# =================================================================================================
# Main Function
# =================================================================================================

def main():
    """Main function - prints deprecation message."""
    log_section("DEPRECATED SCRIPT")
    logger("This script (pipeline_init.py) is no longer used.", "WARNING")
    logger("", "INFO")
    logger("The pipeline now starts from Stage 2 (data ingestion from parquet files),", "INFO")
    logger("not from a raw_lob archive collection.", "INFO")
    logger("", "INFO")
    logger("To initialize and run the pipeline:", "INFO")
    logger("  1. python scripts/02_data_ingestion.py", "INFO")
    logger("  2. python scripts/run_pipeline.py", "INFO")
    logger("", "INFO")
    logger("Or use run_pipeline.py with start_from=2 to run both automatically.", "INFO")
    log_section("", char="=")
    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)