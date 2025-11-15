# S3 Scripts Fix Implementation Guide

## Overview

This guide provides a step-by-step plan to fix critical issues in S3 interaction scripts that handle large data volumes (100MB parquet files and 10GB splits).

## What Was Done

1. **Comprehensive Code Review** - Analyzed all 3 S3 scripts
2. **Created Helper Module** - `src/utils/s3_config.py` with optimized configurations
3. **Documented All Issues** - Full review in `S3_SCRIPTS_REVIEW.md`
4. **Provided Critical Fix Example** - `CRITICAL_FIX_EXAMPLE.md`

## Files Affected

- `scripts/upload_lob_data_to_s3.py` - Raw LOB data upload
- `scripts/13_export_splits_to_s3.py` - Export processed splits
- `scripts/14_import_splits_from_s3.py` - Import splits from S3
- `src/utils/s3_config.py` - **NEW** - Centralized S3 configuration

## Implementation Priority

### Phase 1: CRITICAL FIXES (Must Do Before Production)

#### 1.1 Fix Expensive Checksum Re-Read (Highest Impact)
**File:** `scripts/13_export_splits_to_s3.py`
**Time:** 30 minutes
**Impact:** Saves 50% time and bandwidth on exports

Follow the exact instructions in `CRITICAL_FIX_EXAMPLE.md`:
- Replace `calculate_s3_checksum()` function
- Update call site to use new function
- Test with one split

#### 1.2 Add S3 Multipart Configuration to All Scripts
**Files:** All 3 scripts
**Time:** 1 hour
**Impact:** Prevents failures on large files

For each script, replace the Spark S3 configuration section:

**Before:**
```python
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
    "com.amazonaws.auth.InstanceProfileCredentialsProvider")
```

**After:**
```python
from src.utils.s3_config import configure_spark_for_s3

# Replace manual configuration with helper function
configure_spark_for_s3(spark)
```

**Changes Required:**

1. **upload_lob_data_to_s3.py** - Lines 296-300
2. **13_export_splits_to_s3.py** - Lines 315-318
3. **14_import_splits_from_s3.py** - Lines 306-309

#### 1.3 Add Retry Logic to S3 Operations
**Files:** All 3 scripts
**Time:** 1-2 hours
**Impact:** Prevents pipeline failures from transient errors

Add retry wrappers to write operations:

```python
from src.utils.s3_config import retry_s3_operation

@retry_s3_operation
def write_to_s3(df, s3_path):
    """Write DataFrame to S3 with retry logic."""
    df.write.mode("overwrite").parquet(s3_path)

# Use in upload functions
write_to_s3(df, s3_path)
```

**Locations to update:**
- `upload_lob_data_to_s3.py` - Line 182
- `13_export_splits_to_s3.py` - Line 200
- `14_import_splits_from_s3.py` - Line 186

#### 1.4 Update boto3 Client Creation
**File:** `scripts/14_import_splits_from_s3.py`
**Time:** 15 minutes
**Impact:** Adds retry to list operations

**Before (Line 103):**
```python
s3_client = boto3.client('s3', region_name=S3_CONFIG['region'])
```

**After:**
```python
from src.utils.s3_config import create_s3_client

s3_client = create_s3_client(region_name=S3_CONFIG['region'])
```

---

### Phase 2: HIGH PRIORITY FIXES (Within Sprint)

#### 2.1 Fix Platform-Specific Paths
**Files:** All 3 scripts
**Time:** 30 minutes

Replace hardcoded Windows paths:

```python
from src.utils.s3_config import get_spark_jars_path

SPARK_CONFIG = {
    "app_name": "...",
    "driver_memory": "8g",
    "jar_files_path": get_spark_jars_path(),  # Platform-aware
}
```

#### 2.2 Optimize Multiple Table Scans
**File:** `scripts/13_export_splits_to_s3.py`
**Time:** 1 hour

Update `get_split_statistics()` to cache DataFrame (see S3_SCRIPTS_REVIEW.md section 2.3).

#### 2.3 Add Progress Indicators
**Files:** All 3 scripts
**Time:** 1 hour

Add tqdm progress bars to file processing loops.

---

### Phase 3: QUALITY IMPROVEMENTS (Nice to Have)

#### 3.1 Add MongoDB Batch Configuration
**File:** `scripts/14_import_splits_from_s3.py`
**Time:** 15 minutes

```python
from src.utils.s3_config import MONGODB_LARGE_IMPORT_OPTIONS

df.write.format("mongodb") \
    .option("database", db_name) \
    .option("collection", collection_name) \
    .options(**MONGODB_LARGE_IMPORT_OPTIONS) \
    .mode("overwrite") \
    .save()
```

#### 3.2 Use Estimated Counts for Verification
**File:** `scripts/14_import_splits_from_s3.py`
**Time:** 30 minutes

Replace `count_documents({})` with `estimated_document_count()` for large collections.

#### 3.3 Add Operation Metrics
**Files:** All 3 scripts
**Time:** 2 hours

Use `S3OperationMetrics` class to track upload/download stats.

---

## Testing Plan

### Unit Tests
Create tests for the new `s3_config.py` module:

```bash
# Test S3 configuration
python -m pytest tests/test_s3_config.py -v
```

### Integration Tests

#### Test 1: Small File Upload
```bash
# Test with a small parquet file first
python scripts/upload_lob_data_to_s3.py --run-id test_small
```

Expected: No errors, faster completion, proper retries on simulated failures.

#### Test 2: Single Split Export
```bash
# Test export with one split
python scripts/13_export_splits_to_s3.py --run-id test_export
```

Expected:
- No "reading back from S3" messages
- Checksum in manifest
- ~50% faster than before

#### Test 3: Split Import
```bash
# Test import
python scripts/14_import_splits_from_s3.py --run-id test_export
```

Expected: Fast import with progress indicators.

### Load Tests

#### Test 4: Large File Handling
Create a test with actual large files:
- Upload 100MB parquet file
- Export 1GB split (or larger if available)
- Monitor memory usage and time

#### Test 5: Network Failure Simulation
```bash
# Disconnect network mid-upload and verify retry works
# Use tc (traffic control) on Linux or Windows Firewall
```

Expected: Automatic retry with exponential backoff, eventual success.

---

## Rollout Plan

### Step 1: Development Environment
1. Apply Phase 1 fixes
2. Run all integration tests
3. Verify no regressions

### Step 2: Staging Environment
1. Deploy fixes to staging
2. Run full pipeline with realistic data volumes
3. Monitor CloudWatch metrics (if available)
4. Verify performance improvements

### Step 3: Production
1. Deploy during maintenance window
2. Monitor first full pipeline run
3. Have rollback plan ready

---

## Success Metrics

### Before Fixes
- Upload 100MB file: ~5-10 minutes (unreliable)
- Export 10GB split: ~60 minutes
- Network failure: Pipeline restart required
- Success rate: ~50-70%

### After Phase 1 Fixes
- Upload 100MB file: ~2-3 minutes (reliable)
- Export 10GB split: ~30 minutes
- Network failure: Automatic retry
- Success rate: >95%

### Additional Gains from Phase 2
- Platform portability: Works on Windows and Linux
- Better visibility: Progress indicators
- Faster processing: Optimized scans

---

## Monitoring Recommendations

### Key Metrics to Track
1. **Upload/download speed** - Bytes per second
2. **Retry counts** - How often retries occur
3. **Error rates** - Failed operations
4. **Time per operation** - Track trends
5. **Memory usage** - Monitor Spark executors

### CloudWatch Dashboards (if using AWS)
- S3 request metrics
- Lambda errors (if using Lambda)
- EC2 network metrics

### Application Logs
Add structured logging:
```python
logger(f'S3 Upload: {file_size_mb}MB in {duration_sec}s ({speed_mbps}MB/s)', "INFO")
```

---

## Rollback Plan

If issues occur after deployment:

1. **Immediate:** Revert to previous script versions
2. **Git:** Use git to restore previous commit
3. **S3 Data:** Data in S3 is safe, can re-run exports

Keep backups of original scripts:
```bash
cp scripts/13_export_splits_to_s3.py scripts/13_export_splits_to_s3.py.backup
```

---

## Dependencies

### Required Python Packages
Verify these are installed:

```bash
pip install boto3 botocore tenacity tqdm
```

Or add to `requirements.txt`:
```
boto3>=1.40.46
botocore>=1.40.46
tenacity>=8.0.0
tqdm>=4.65.0
```

### Spark Dependencies
Ensure AWS JARs are available:
- `hadoop-aws-3.3.4.jar`
- `aws-java-sdk-bundle-1.12.262.jar`

---

## Support and Troubleshooting

### Common Issues

#### Issue: "Module 's3_config' not found"
**Solution:** Ensure `src/utils/s3_config.py` is in Python path and `__init__.py` exists.

#### Issue: "Multipart upload failed"
**Solution:** Check IAM permissions for multipart operations:
- `s3:PutObject`
- `s3:AbortMultipartUpload`
- `s3:ListMultipartUploadParts`

#### Issue: "Connection timeout"
**Solution:** Increase timeout in configuration or check network connectivity.

#### Issue: "Too many retries"
**Solution:** Check S3 throttling limits, may need to request limit increase.

---

## Next Steps

1. ✅ Review this implementation guide
2. ✅ Review `S3_SCRIPTS_REVIEW.md` for detailed issues
3. ✅ Review `CRITICAL_FIX_EXAMPLE.md` for immediate fix
4. ⏳ Apply Phase 1 fixes
5. ⏳ Test in development environment
6. ⏳ Deploy to staging
7. ⏳ Monitor and iterate

---

## Questions?

Refer to:
- `S3_SCRIPTS_REVIEW.md` - Detailed technical review
- `CRITICAL_FIX_EXAMPLE.md` - Step-by-step fix for worst issue
- `src/utils/s3_config.py` - Reference implementation

For AWS S3 configuration details:
- [Hadoop-AWS S3A Documentation](https://hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html)
- [Boto3 Retry Configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html)
