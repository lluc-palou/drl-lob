"""
VQ-VAE Production Training Script (Stage 16)

TRAIN MODE: Trains VQ-VAE models per split using role='train' data
TEST MODE: Trains VQ-VAE on full split_0 (train+val) and encodes test_data

Trains final VQ-VAE models using best hyperparameters from Stage 15.

Train Mode:
- Input: split_X_input collections (role='train')
- Output: split_X_model.pth + encoded split_X_input (all roles)
- Saves to: artifacts/vqvae_models/production/

Test Mode:
- Input: split_0_input (all roles), test_data
- Output: split_0_full_model.pth + encoded test_data
- Saves to: artifacts/vqvae_models/test/

Usage:
    TRAIN: python scripts/14_vqvae_production.py --mode train
    TRAIN: python scripts/14_vqvae_production.py --mode train --splits even
    TEST:  python scripts/14_vqvae_production.py --mode test --test-split 0
"""

import os
import sys
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# Unicode/MLflow Fix for Windows - MUST BE FIRST!
# =================================================================================================
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
    os.environ['PYTHONUTF8'] = '1'
    
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

# Patch MLflow emoji issue
try:
    from mlflow.tracking._tracking_service import client as mlflow_client
    
    _original_log_url = mlflow_client.TrackingServiceClient._log_url
    
    def _patched_log_url(self, run_id):
        try:
            run = self.get_run(run_id)
            run_name = run.info.run_name or run_id
            run_url = self._get_run_url(run.info.experiment_id, run_id)
            sys.stdout.write(f"[RUN] View run {run_name} at: {run_url}\n")
            sys.stdout.flush()
        except:
            pass
    
    mlflow_client.TrackingServiceClient._log_url = _patched_log_url
except:
    pass
# =================================================================================================

import json
import yaml
import torch
import mlflow
from datetime import datetime

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.vqvae_representation import (
    run_production_training,
    TRAINING_CONFIG
)

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"  # Read from feature standardization output (Stage 11)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "VQ-VAE_Production_Training"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "8g"

# Artifact directories
ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts" / "vqvae_models"
HYPERPARAMETER_SEARCH_DIR = ARTIFACT_BASE_DIR / "hyperparameter_search"
PRODUCTION_DIR = ARTIFACT_BASE_DIR / "production"

# =================================================================================================
# Helper Functions
# =================================================================================================

def load_best_config() -> dict:
    """
    Load best hyperparameter configuration from Stage 12.

    Returns:
        Best configuration dictionary
    """
    config_path = HYPERPARAMETER_SEARCH_DIR / "best_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Best config not found at {config_path}. "
            f"Please run Stage 12 (hyperparameter search) first."
        )

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    logger(f'Loaded best config from: {config_path}', "INFO")
    logger(f'  Average validation loss: {config_data["metrics"]["avg_val_loss"]:.4f}', "INFO")
    logger(f'  Configuration: {config_data["best_config"]}', "INFO")

    return config_data["best_config"]

# =================================================================================================
# Main Execution
# =================================================================================================

def filter_splits(all_splits: list, split_filter: str = None) -> list:
    """
    Filter splits based on filter specification.

    Args:
        all_splits: List of all available split IDs
        split_filter: Filter specification ('even', 'odd', or comma-separated list)

    Returns:
        Filtered list of split IDs
    """
    if not split_filter:
        return all_splits

    split_filter = split_filter.lower().strip()

    if split_filter == 'even':
        filtered = [s for s in all_splits if s % 2 == 0]
        logger(f'Filtering even splits: {filtered}', "INFO")
        return filtered
    elif split_filter == 'odd':
        filtered = [s for s in all_splits if s % 2 == 1]
        logger(f'Filtering odd splits: {filtered}', "INFO")
        return filtered
    else:
        # Comma-separated list of specific splits
        try:
            specific_splits = [int(s.strip()) for s in split_filter.split(',')]
            filtered = [s for s in all_splits if s in specific_splits]
            logger(f'Filtering specific splits: {filtered}', "INFO")
            return filtered
        except ValueError:
            raise ValueError(f"Invalid split filter: {split_filter}. Use 'even', 'odd', or comma-separated numbers")

def main(mode: str = 'train', test_split: int = 0, split_filter: str = None):
    """Main execution function.

    Args:
        mode: Pipeline mode ('train' or 'test')
        test_split: Split ID to use for full training in test mode
        split_filter: Optional filter for splits in train mode ('even', 'odd', or comma-separated list)
    """
    logger('=' * 100, "INFO")
    logger(f'VQ-VAE PRODUCTION TRAINING (STAGE 16) - {mode.upper()} MODE', "INFO")
    logger('=' * 100, "INFO")

    if mode == 'train' and split_filter:
        logger(f'Split filter: {split_filter}', "INFO")
        logger('', "INFO")
    elif mode == 'test':
        logger(f'Training on full split_{test_split} (train+val combined)', "INFO")
        logger(f'Will encode test_data collection', "INFO")
        logger('', "INFO")
    
    # Load best configuration
    logger('', "INFO")
    logger('Loading best hyperparameter configuration from Stage 12...', "INFO")
    best_config = load_best_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger('', "INFO")
    logger(f'Device: {device}', "INFO")
    
    if device.type == 'cuda':
        logger(f'CUDA Device: {torch.cuda.get_device_name(0)}', "INFO")
        logger(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB', "INFO")
    
    # Setup MLflow
    logger('', "INFO")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}', "INFO")
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark session
    logger('', "INFO")
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="VQVAEProductionTraining",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Display configuration
        logger('', "INFO")
        logger('Production training configuration:', "INFO")
        logger(f'  Database: {DB_NAME}', "INFO")
        logger(f'  Collection pattern: {COLLECTION_PREFIX}*{COLLECTION_SUFFIX}', "INFO")
        logger(f'  Production model directory: {PRODUCTION_DIR}', "INFO")
        
        logger('', "INFO")
        logger('Best hyperparameter configuration:', "INFO")
        for key, value in best_config.items():
            logger(f'  {key}: {value}', "INFO")
        
        logger('', "INFO")
        logger('Training configuration:', "INFO")
        for key, value in TRAINING_CONFIG.items():
            logger(f'  {key}: {value}', "INFO")

        # CRITICAL: Create timestamp indexes on all split collections for efficient hourly queries
        # Without these indexes, each hourly query performs a full collection scan O(N)
        # With indexes: O(log N + matches) - reduces processing time dramatically
        logger('', "INFO")
        logger('Creating timestamp indexes on all split collections...', "INFO")
        from pymongo import MongoClient, ASCENDING
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        all_collections = db.list_collection_names()

        # Find all split collections
        split_collections = [c for c in all_collections
                           if c.startswith(COLLECTION_PREFIX) and c.endswith(COLLECTION_SUFFIX)]

        for collection_name in split_collections:
            input_coll = db[collection_name]

            # Check if index already exists
            existing_indexes = list(input_coll.list_indexes())
            has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

            if not has_timestamp_index:
                logger(f'  Creating index on {collection_name}...', "INFO")
                input_coll.create_index([("timestamp", ASCENDING)], background=False)
            else:
                logger(f'  Index already exists on {collection_name}', "INFO")

        client.close()
        logger(f'Timestamp indexes created/verified on {len(split_collections)} collections', "INFO")

        if mode == 'train':
            # TRAIN MODE: Train models per split
            # Discover and filter splits if needed
            from src.vqvae_representation.data_loader import discover_splits
            all_split_ids = discover_splits(spark, DB_NAME, COLLECTION_PREFIX, COLLECTION_SUFFIX)
            filtered_split_ids = filter_splits(all_split_ids, split_filter) if split_filter else None

            # Run production training
            logger('', "INFO")
            logger('Starting production training...', "INFO")
            logger('', "INFO")

            results = run_production_training(
                spark=spark,
                db_name=DB_NAME,
                collection_prefix=COLLECTION_PREFIX,
                collection_suffix=COLLECTION_SUFFIX,
                device=device,
                best_config=best_config,
                mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
                production_dir=PRODUCTION_DIR,
                mongo_uri=MONGO_URI,
                use_pymongo=True,
                split_ids_filter=filtered_split_ids
            )

            # Summary
            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger('PRODUCTION TRAINING COMPLETE (TRAIN MODE)', "INFO")
            logger('=' * 100, "INFO")
            logger(f'Models trained: {results["num_splits"]}', "INFO")
            logger(f'Total samples processed: {results["total_samples"]:,}', "INFO")
            logger(f'Average validation loss: {results["avg_val_loss"]:.4f}', "INFO")
            logger(f'Models saved to: {PRODUCTION_DIR}', "INFO")
            logger(f'Latent collections: {COLLECTION_PREFIX}*_input (renamed)', "INFO")

        else:  # TEST MODE
            # TEST MODE: Train on full split, encode test_data
            test_production_dir = ARTIFACT_BASE_DIR / "test"
            test_production_dir.mkdir(parents=True, exist_ok=True)

            logger('', "INFO")
            logger('Starting test mode training...', "INFO")
            logger(f'Training VQ-VAE on full split_{test_split} (train+val combined)', "INFO")
            logger(f'Will encode test_data collection', "INFO")
            logger('', "INFO")

            # NOTE: Test mode requires modifications to run_production_training to:
            # 1. Train on split with no role filter (train+val combined)
            # 2. Save model as split_{test_split}_full_model.pth
            # 3. Encode test_data collection instead of split collections
            # This implementation provides the CLI structure; actual training logic
            # needs to be implemented in src.vqvae_representation.run_production_training

            results = run_production_training(
                spark=spark,
                db_name=DB_NAME,
                collection_prefix=COLLECTION_PREFIX,
                collection_suffix=COLLECTION_SUFFIX,
                device=device,
                best_config=best_config,
                mlflow_experiment_name=f"{MLFLOW_EXPERIMENT_NAME}_Test",
                production_dir=test_production_dir,
                mongo_uri=MONGO_URI,
                use_pymongo=True,
                split_ids_filter=[test_split],  # Only train on test_split
                test_mode=True,  # Flag to indicate test mode
                test_collection='test_data'  # Encode test_data instead of split
            )

            # Summary
            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger('PRODUCTION TRAINING COMPLETE (TEST MODE)', "INFO")
            logger('=' * 100, "INFO")
            logger(f'Model trained on: split_{test_split} (full data)', "INFO")
            logger(f'Model saved to: {test_production_dir}', "INFO")
            logger(f'test_data collection encoded with codebook indices', "INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        
        if mlflow.active_run():
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_message", str(e))
            mlflow.end_run(status="FAILED")
        
        raise
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='VQ-VAE Production Training (Stage 16)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train mode - all splits:
  python scripts/14_vqvae_production.py --mode train

  # Train mode - even splits only:
  python scripts/14_vqvae_production.py --mode train --splits even

  # Test mode - train on full split_0, encode test_data:
  python scripts/14_vqvae_production.py --mode test --test-split 0
        """
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        default='train',
        help='Pipeline mode: train (per-split models) or test (full split + test_data)'
    )
    parser.add_argument(
        '--test-split',
        type=int,
        default=0,
        help='Split ID to use for full training in test mode (default: 0)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        default=None,
        help='[TRAIN MODE ONLY] Filter splits: "even", "odd", or comma-separated list (e.g., "0,1,2")'
    )
    args = parser.parse_args()

    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'

    import time
    start_time = time.time()

    try:
        main(mode=args.mode, test_split=args.test_split, split_filter=args.splits)

        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logger('', "INFO")
        logger(f'Total execution time: {hours}h {minutes}m {seconds}s', "INFO")
        logger(f'Stage 16 completed successfully ({args.mode} mode)', "INFO")

    except Exception:
        sys.exit(1)