"""
MLflow Windows UTF-8 Patch Utility

Fixes Windows console encoding issues and MLflow emoji errors.
This utility should be imported at the top of scripts that use MLflow.

Usage:
    from src.utils.mlflow_patch import apply_mlflow_patch
    apply_mlflow_patch()  # Call once at script startup
"""

import sys
import os


def apply_mlflow_patch():
    """
    Apply Windows UTF-8 fix and MLflow emoji patch.

    This function:
    1. Configures Windows console for UTF-8 encoding (fixes emoji errors)
    2. Patches MLflow tracking service client to remove emoji from output
    3. Handles errors gracefully (won't crash if MLflow isn't imported yet)

    Should be called early in script execution, before other imports.
    """

    # =================================================================================================
    # Unicode/MLflow Fix for Windows - MUST BE FIRST!
    # =================================================================================================
    # Fix Windows console encoding to handle Unicode characters (fixes MLflow emoji errors)
    if sys.platform == 'win32':
        # Set environment variables for UTF-8 support
        os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
        os.environ['PYTHONUTF8'] = '1'

        # Reconfigure stdout/stderr to use UTF-8 with error replacement
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass  # Silently fail if reconfiguration doesn't work

    # Patch MLflow to remove emoji that causes Windows encoding errors
    try:
        from mlflow.tracking._tracking_service import client as mlflow_client

        _original_log_url = mlflow_client.TrackingServiceClient._log_url

        def _patched_log_url(self, run_id):
            """Patched MLflow URL logger - replaces emoji with [RUN]"""
            try:
                run = self.get_run(run_id)
                run_name = run.info.run_name or run_id
                run_url = self._get_run_url(run.info.experiment_id, run_id)
                sys.stdout.write(f"[RUN] View run {run_name} at: {run_url}\n")
                sys.stdout.flush()
            except Exception:
                pass  # Silently skip if anything fails

        mlflow_client.TrackingServiceClient._log_url = _patched_log_url
    except Exception:
        pass  # MLflow not imported yet or patch failed - that's OK


# Automatically apply patch when module is imported
# This allows simple usage: from src.utils.mlflow_patch import apply_mlflow_patch
apply_mlflow_patch()
