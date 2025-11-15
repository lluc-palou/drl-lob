# Critical Fix Example: Remove Expensive Checksum Re-Read

## Problem

In `scripts/13_export_splits_to_s3.py`, the `calculate_s3_checksum()` function downloads the ENTIRE file back from S3 just to calculate a checksum. For 10GB splits, this means:

1. Upload 10GB to S3 (15-30 minutes)
2. Download 10GB back from S3 (15-30 minutes)
3. Scan all 10GB to count rows (5-10 minutes)

**Total waste per split: 35-70 minutes and double bandwidth usage!**

## Solution

Calculate checksum BEFORE upload using metadata we already have, or use S3's built-in ETag.

## Code Changes for 13_export_splits_to_s3.py

### Option 1: Use Metadata-Based Checksum (Recommended)

Replace the entire `calculate_s3_checksum()` function and update how it's called:

```python
# BEFORE (Lines 214-231) - DELETE THIS:
def calculate_s3_checksum(spark: SparkSession, s3_path: str) -> str:
    """
    Calculate checksum for S3 Parquet files.
    This is a simple implementation - in production you might want to use S3 ETags.
    """
    try:
        # Read back from S3 and calculate basic checksum on document count + schema
        df = spark.read.parquet(s3_path)
        count = df.count()
        schema_str = str(df.schema)
        checksum_input = f"{count}:{schema_str}"
        checksum = hashlib.md5(checksum_input.encode()).hexdigest()
        return f"md5:{checksum}"
    except Exception as e:
        logger(f'Warning: Could not calculate checksum: {e}', "WARNING")
        return "unknown"


# AFTER - REPLACE WITH THIS:
def calculate_split_checksum_from_stats(stats: Dict[str, Any]) -> str:
    """
    Calculate checksum from pre-computed statistics.
    Much more efficient than re-reading from S3.

    Args:
        stats: Dictionary with split statistics (from get_split_statistics)

    Returns:
        MD5 checksum string
    """
    try:
        # Create checksum from metadata
        checksum_components = [
            str(stats.get("num_documents", 0)),
            str(stats.get("schema_fields", [])),
            str(stats.get("feature_length", 0)),
            str(sorted(stats.get("role_distribution", {}).items())),
        ]
        checksum_input = ":".join(checksum_components)
        checksum = hashlib.md5(checksum_input.encode()).hexdigest()
        return f"md5:{checksum}"
    except Exception as e:
        logger(f'Warning: Could not calculate checksum: {e}', "WARNING")
        return "unknown"
```

Then update the export function (Lines 163-212):

```python
# BEFORE (Lines 359-362):
if stats:
    # Calculate checksum
    stats["checksum"] = calculate_s3_checksum(spark, s3_path)
    split_stats.append(stats)

# AFTER - REPLACE WITH:
if stats:
    # Calculate checksum from stats (no S3 re-read needed!)
    stats["checksum"] = calculate_split_checksum_from_stats(stats)
    split_stats.append(stats)
```

### Option 2: Use S3 ETag (Alternative)

Or use S3's built-in ETag which is automatically calculated during upload:

```python
from src.utils.s3_config import get_s3_etag

# After successful upload (Line 361):
if stats:
    # Get S3 ETag (no download needed!)
    try:
        stats["checksum"] = f"etag:{get_s3_etag(s3_path, S3_CONFIG['region'])}"
    except Exception as e:
        logger(f'Warning: Could not get S3 ETag: {e}', "WARNING")
        stats["checksum"] = calculate_split_checksum_from_stats(stats)
    split_stats.append(stats)
```

## Expected Impact

### Before Fix:
- **Time per 10GB split**: ~60 minutes (upload 30min + download 30min)
- **Bandwidth**: 20GB per split (10GB up + 10GB down)
- **For 10 splits**: 600 minutes (10 hours), 200GB bandwidth

### After Fix:
- **Time per 10GB split**: ~30 minutes (upload only)
- **Bandwidth**: 10GB per split (upload only)
- **For 10 splits**: 300 minutes (5 hours), 100GB bandwidth

**Savings: 50% time reduction, 50% bandwidth reduction!**

## Testing

After applying the fix, test with a small split:

```bash
# Export a single split
python scripts/13_export_splits_to_s3.py --run-id test_fix

# Verify:
# 1. No "Reading back from S3" messages in logs
# 2. Checksum is in manifest
# 3. Export time is approximately upload time only
```

## Related Fixes

This is the MOST CRITICAL fix. After applying this:

1. Apply S3 configuration fixes (see `src/utils/s3_config.py`)
2. Add retry logic to upload operations
3. Optimize table scans (use caching)

See `S3_SCRIPTS_REVIEW.md` for complete list of issues and fixes.
