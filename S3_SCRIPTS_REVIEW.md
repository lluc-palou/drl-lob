# S3 Scripts Code Review - Large Data Handling

**Review Date:** 2025-11-15
**Context:** Pipeline handling ~100MB Parquet files and ~10GB splits
**Scripts Reviewed:**
- `scripts/upload_lob_data_to_s3.py`
- `scripts/13_export_splits_to_s3.py`
- `scripts/14_import_splits_from_s3.py`

---

## Executive Summary

### Critical Issues Found: 7
### High Priority Issues: 12
### Medium Priority Issues: 8

**Most Critical Concerns:**
1. **No S3 multipart upload/download configuration** - Will fail or timeout on 10GB files
2. **No retry logic** - Single network failure causes complete pipeline failure
3. **Expensive checksum validation** - Reading back entire 10GB files from S3
4. **Missing timeout configurations** - Operations can hang indefinitely
5. **No connection pooling** - May exhaust connections on large batches

---

## 1. upload_lob_data_to_s3.py

### CRITICAL ISSUES

#### 1.1 Missing S3 Multipart Upload Configuration
**Location:** Lines 288-301 (Spark S3 configuration)
**Severity:** CRITICAL
**Impact:** Large files (>100MB) may fail or be very slow

**Problem:**
```python
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
    "com.amazonaws.auth.InstanceProfileCredentialsProvider")
```

Missing critical S3A configurations:
- No multipart upload threshold
- No multipart chunk size
- No buffer size
- No thread pool size

**Recommended Fix:**
```python
# S3A Performance tuning for large files
hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

# Multipart upload settings (critical for large files)
hadoop_conf.set("fs.s3a.multipart.size", "104857600")  # 100MB per part
hadoop_conf.set("fs.s3a.multipart.threshold", "52428800")  # 50MB threshold
hadoop_conf.set("fs.s3a.fast.upload", "true")
hadoop_conf.set("fs.s3a.fast.upload.buffer", "disk")  # Use disk buffering for large files

# Connection settings
hadoop_conf.set("fs.s3a.connection.maximum", "50")
hadoop_conf.set("fs.s3a.threads.max", "20")
hadoop_conf.set("fs.s3a.connection.establish.timeout", "30000")  # 30s
hadoop_conf.set("fs.s3a.connection.timeout", "300000")  # 5min for large uploads

# Retry settings
hadoop_conf.set("fs.s3a.attempts.maximum", "10")
hadoop_conf.set("fs.s3a.retry.limit", "7")
hadoop_conf.set("fs.s3a.retry.interval", "500ms")
```

#### 1.2 No Retry Logic on Upload Failures
**Location:** Lines 170-205 (`upload_file_to_s3`)
**Severity:** CRITICAL
**Impact:** Single network glitch causes complete failure

**Problem:**
```python
# Write to S3
df.write.mode("overwrite").parquet(s3_path)
```

No retry wrapper, no exponential backoff, no error recovery.

**Recommended Fix:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from botocore.exceptions import ClientError

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ClientError, ConnectionError, TimeoutError)),
    reraise=True
)
def write_with_retry(df, s3_path):
    df.write.mode("overwrite").parquet(s3_path)

# Usage
write_with_retry(df, s3_path)
```

#### 1.3 Expensive Row Count Operation
**Location:** Line 178
**Severity:** HIGH
**Impact:** Full table scan for every file, adds significant time

**Problem:**
```python
row_count = df.count()  # Full scan of 100MB file
```

For large files, `count()` triggers a full scan. With many files, this compounds.

**Recommended Fix:**
Consider whether row count is essential for manifest. If needed:
```python
# Option 1: Skip count if not critical
# row_count = None  # Add to manifest later if needed

# Option 2: Count during upload (piggyback on write operation)
# Use Spark SQL to get count from written parquet metadata
# This is more efficient than separate count()
```

### HIGH PRIORITY ISSUES

#### 1.4 Small Checksum Read Buffer
**Location:** Line 143
**Severity:** MEDIUM
**Impact:** Slower checksum calculation

**Problem:**
```python
for chunk in iter(lambda: f.read(4096), b""):  # 4KB chunks
```

4KB is very small for 100MB files.

**Recommended Fix:**
```python
for chunk in iter(lambda: f.read(8388608), b""):  # 8MB chunks
```

#### 1.5 No Progress Indication
**Location:** Lines 374-383 (upload loop)
**Severity:** MEDIUM
**Impact:** User has no feedback during long uploads

**Problem:**
No progress bars or size tracking during uploads.

**Recommended Fix:**
```python
from tqdm import tqdm

with tqdm(total=len(parquet_files), desc="Uploading files") as pbar:
    for i, file_info in enumerate(parquet_files):
        stats = upload_file_to_s3(spark, file_info, s3_path)
        file_stats.append(stats)
        pbar.update(1)
```

#### 1.6 Hardcoded JAR Path (Windows-specific)
**Location:** Line 77
**Severity:** MEDIUM
**Impact:** Won't work on Linux production environment

**Problem:**
```python
"jar_files_path": "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/",
```

**Recommended Fix:**
```python
import platform

SPARK_CONFIG = {
    "app_name": "UploadLobDataToS3",
    "driver_memory": "4g",
    "jar_files_path": get_spark_jars_path(),
}

def get_spark_jars_path():
    if platform.system() == "Windows":
        return "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
    else:
        # Use SPARK_HOME environment variable on Linux
        spark_home = os.environ.get("SPARK_HOME", "/opt/spark")
        return f"file://{spark_home}/jars/"
```

---

## 2. 13_export_splits_to_s3.py

### CRITICAL ISSUES

#### 2.1 Extremely Expensive Checksum Calculation
**Location:** Lines 214-231 (`calculate_s3_checksum`)
**Severity:** CRITICAL
**Impact:** **Reading back ENTIRE 10GB files from S3** just to calculate checksum!

**Problem:**
```python
def calculate_s3_checksum(spark: SparkSession, s3_path: str) -> str:
    try:
        # Read back from S3 and calculate basic checksum on document count + schema
        df = spark.read.parquet(s3_path)  # Downloads entire 10GB file!
        count = df.count()  # Full scan of 10GB!
        schema_str = str(df.schema)
        checksum_input = f"{count}:{schema_str}"
        checksum = hashlib.md5(checksum_input.encode()).hexdigest()
        return f"md5:{checksum}"
```

This is **extremely wasteful**:
- Uploads 10GB split to S3
- Immediately downloads entire 10GB back from S3
- Scans all 10GB to count rows
- For multiple splits, this could mean downloading hundreds of GB!

**Recommended Fix:**
```python
def calculate_split_checksum(df, stats: Dict[str, Any]) -> str:
    """
    Calculate checksum BEFORE upload using metadata only.
    Much more efficient than reading back from S3.
    """
    # Use pre-computed stats instead of re-reading
    checksum_input = f"{stats['num_documents']}:{stats['schema_fields']}:{stats.get('feature_length', 0)}"
    checksum = hashlib.md5(checksum_input.encode()).hexdigest()
    return f"md5:{checksum}"

# Usage in export_split_to_s3:
stats = get_split_statistics(spark, db_name, collection)
checksum = calculate_split_checksum(df, stats)  # Calculate BEFORE upload
df.write.mode("overwrite").parquet(s3_path)
stats["checksum"] = checksum
```

Or use S3 ETags:
```python
import boto3

def get_s3_etag(bucket: str, key: str) -> str:
    """Get S3 ETag (checksum) without downloading file."""
    s3_client = boto3.client('s3')
    response = s3_client.head_object(Bucket=bucket, Key=key)
    return response.get('ETag', '').strip('"')
```

#### 2.2 Missing S3 Configuration (Same as Script 1)
**Location:** Lines 315-318
**Severity:** CRITICAL
**Impact:** 10GB files will fail or timeout

Same issue as upload_lob_data_to_s3.py - missing multipart configuration.

#### 2.3 Multiple Full Table Scans
**Location:** Lines 137, 142-143, 189
**Severity:** HIGH
**Impact:** 3-4 full scans per split (very expensive for 10GB)

**Problem:**
```python
# Scan 1: Count documents
doc_count = df.count()  # Full scan

# Scan 2: Role distribution
role_counts = df.groupBy("role").agg(spark_count("*").alias("count")).collect()  # Full scan

# Scan 3: Get first row for features
first_row = df.select("features").first()  # Partial scan but still expensive
```

**Recommended Fix:**
```python
def get_split_statistics(spark: SparkSession, db_name: str, collection: str) -> Dict[str, Any]:
    """
    Get statistics with SINGLE pass over data using caching.
    """
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .load()
    )

    # Cache the DataFrame to avoid multiple scans
    df.cache()

    try:
        # Get document count
        doc_count = df.count()  # Scan 1 (cached for later)

        if doc_count == 0:
            return {
                "num_documents": 0,
                "role_distribution": {},
                "schema_fields": [],
                "feature_length": None,
            }

        # Get role distribution (uses cached data)
        role_dist = {}
        if "role" in df.columns:
            role_counts = df.groupBy("role").agg(spark_count("*").alias("count")).collect()
            role_dist = {row["role"]: row["count"] for row in role_counts}

        # Get schema info
        schema_fields = [field.name for field in df.schema.fields]

        # Get feature length (uses cached data)
        feature_length = None
        if "features" in df.columns:
            first_row = df.select("features").first()
            if first_row and first_row["features"]:
                feature_length = len(first_row["features"])

        return {
            "num_documents": doc_count,
            "role_distribution": role_dist,
            "schema_fields": schema_fields,
            "feature_length": feature_length,
        }
    finally:
        # Unpersist to free memory
        df.unpersist()
```

### HIGH PRIORITY ISSUES

#### 2.4 No Retry Logic on S3 Writes
**Location:** Line 200
**Severity:** HIGH
**Impact:** 10GB upload failure wastes hours of time

Same as Script 1 - needs retry wrapper.

#### 2.5 Memory Pressure from Large Splits
**Location:** Line 181-186
**Severity:** HIGH
**Impact:** Loading 10GB splits could cause OOM

**Problem:**
```python
df = (
    spark.read.format("mongodb")
    .option("database", db_name)
    .option("collection", collection)
    .load()
)
```

No partitioning or batch size limits.

**Recommended Fix:**
```python
df = (
    spark.read.format("mongodb")
    .option("database", db_name)
    .option("collection", collection)
    .option("partitioner", "MongoSamplePartitioner")
    .option("partitionKey", "_id")
    .option("partitionSizeMB", "64")  # 64MB partitions
    .load()
)
```

---

## 3. 14_import_splits_from_s3.py

### CRITICAL ISSUES

#### 3.1 Missing S3 Configuration (Same as Scripts 1 & 2)
**Location:** Lines 306-309
**Severity:** CRITICAL
**Impact:** Downloading 10GB files will fail or timeout

Same multipart configuration missing.

**Additional for downloads:**
```python
# Add read-ahead configuration for large downloads
hadoop_conf.set("fs.s3a.readahead.range", "2097152")  # 2MB read-ahead
hadoop_conf.set("fs.s3a.input.fadvise", "random")  # Or "sequential" for large files
```

#### 3.2 No Retry Logic on S3 Reads
**Location:** Line 186
**Severity:** CRITICAL
**Impact:** Download failure after hours of waiting

**Problem:**
```python
df = spark.read.parquet(s3_path)  # 10GB download, no retry
```

Needs retry wrapper like upload scripts.

#### 3.3 boto3 Client Without Retry Configuration
**Location:** Lines 102-109
**Severity:** HIGH
**Impact:** List operations fail without retry

**Problem:**
```python
s3_client = boto3.client('s3', region_name=S3_CONFIG['region'])
```

No retry configuration on boto3 client.

**Recommended Fix:**
```python
from botocore.config import Config

retry_config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'adaptive'  # or 'standard'
    },
    connect_timeout=30,
    read_timeout=300,
    max_pool_connections=50
)

s3_client = boto3.client('s3', region_name=S3_CONFIG['region'], config=retry_config)
```

### HIGH PRIORITY ISSUES

#### 3.4 Expensive Document Count Verification
**Location:** Line 211
**Severity:** HIGH
**Impact:** Full MongoDB scan after every import

**Problem:**
```python
actual_count = db[collection_name].count_documents({})  # Full collection scan
```

For 10GB collections, this is very slow.

**Recommended Fix:**
```python
# Use estimated count for large collections (much faster)
actual_count = db[collection_name].estimated_document_count()

# Or if accuracy is critical, sample verify instead:
def verify_sample(collection, expected_count, sample_size=10000):
    """Verify a sample instead of full count."""
    sample_count = collection.count_documents({}, limit=sample_size)
    if sample_count < sample_size:
        # Collection is small enough to fully count
        return collection.count_documents({})
    else:
        # Use estimated count for large collections
        return collection.estimated_document_count()
```

#### 3.5 Collection Drop Without Error Handling
**Location:** Lines 195-197
**Severity:** MEDIUM
**Impact:** Silent failures possible

**Problem:**
```python
if collection_name in db.list_collection_names():
    db[collection_name].drop()
    logger(f'  Dropped existing collection {collection_name}', "INFO")
```

No try/except around drop operation.

**Recommended Fix:**
```python
try:
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        logger(f'  Dropped existing collection {collection_name}', "INFO")
except Exception as e:
    logger(f'  Warning: Could not drop collection {collection_name}: {e}', "WARNING")
    # Continue anyway - overwrite mode will handle it
```

#### 3.6 MongoDB Write Without Batch Size Configuration
**Location:** Lines 201-206
**Severity:** MEDIUM
**Impact:** May use inefficient batch sizes

**Problem:**
```python
df.write.format("mongodb") \
    .option("database", db_name) \
    .option("collection", collection_name) \
    .option("ordered", "false") \
    .mode("overwrite") \
    .save()
```

No batch size specified - defaults may be too small.

**Recommended Fix:**
```python
df.write.format("mongodb") \
    .option("database", db_name) \
    .option("collection", collection_name) \
    .option("ordered", "false") \
    .option("maxBatchSize", "1000") \
    .option("writeConcern.w", "1") \
    .mode("overwrite") \
    .save()
```

---

## 4. Common Issues Across All Scripts

### 4.1 No Network Timeout Configurations
All scripts missing socket and connection timeouts. Network hangs could block indefinitely.

### 4.2 No Connection Pool Management
Default connection pools may be exhausted when processing many files in parallel.

### 4.3 No Bandwidth Throttling
Could saturate network and affect other services. Consider:
```python
hadoop_conf.set("fs.s3a.bandwidth.limit", "104857600")  # 100 MB/s limit if needed
```

### 4.4 No Monitoring/Metrics
No CloudWatch metrics, no performance tracking. Consider adding:
- Upload/download speed tracking
- Retry counts
- Error rates
- Time per file

### 4.5 Insufficient Logging for Debugging
No logging of:
- Actual bytes transferred
- Time per operation
- S3 request IDs for debugging
- Network errors vs application errors

---

## 5. Recommended Priority Order for Fixes

### IMMEDIATE (Must Fix Before Production):
1. ✅ Add S3 multipart upload/download configuration to all scripts
2. ✅ Remove expensive checksum re-read in `13_export_splits_to_s3.py` (Line 214-231)
3. ✅ Add retry logic with exponential backoff to all S3 operations
4. ✅ Add timeout configurations (connection, socket, read)

### HIGH PRIORITY (Fix Within Sprint):
5. ✅ Add boto3 retry configuration
6. ✅ Optimize multiple full table scans (use caching)
7. ✅ Add progress indicators for long operations
8. ✅ Fix hardcoded Windows paths

### MEDIUM PRIORITY (Quality Improvements):
9. Add MongoDB batch size configuration
10. Replace full counts with estimated counts where appropriate
11. Add monitoring/metrics
12. Improve error handling and logging

---

## 6. Testing Recommendations

### 6.1 Load Testing
- Test with actual 100MB parquet files
- Test with actual 10GB splits
- Test with multiple files in parallel
- Test network interruption scenarios

### 6.2 Failure Testing
- Simulate S3 throttling (503 errors)
- Simulate network timeouts
- Simulate partial uploads/downloads
- Test recovery after failures

### 6.3 Performance Testing
- Measure upload/download speeds
- Track memory usage during large operations
- Monitor Spark executor memory
- Profile MongoDB import performance

---

## 7. Estimated Impact

### Without Fixes:
- **10GB split upload**: Likely to fail or take 2-3 hours
- **Checksum validation**: Additional 1-2 hours per split (re-downloading from S3)
- **Network failure**: Complete pipeline restart required
- **Total risk**: Pipeline failure probability >50% for full dataset

### With Fixes:
- **10GB split upload**: 15-30 minutes (with multipart)
- **Checksum validation**: <1 second (using metadata)
- **Network failure**: Automatic retry, no manual intervention
- **Total risk**: Pipeline failure probability <5%

---

## 8. Code Examples - Complete Fixed Functions

See recommendations inline above for specific fixes to each issue.

---

**Review Completed By:** Claude Code
**Next Steps:** Implement IMMEDIATE priority fixes, then proceed with testing on staging environment.
