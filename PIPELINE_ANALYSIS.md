# DRL-LOB Pipeline Architecture Analysis

## 1. PIPELINE FLOW AND STAGES

The pipeline consists of 14 stages (stages 2-14), organized as follows:

### Stage Overview:
- **Stage 2**: Raw LOB Data Ingestion - Load parquet files to MongoDB
- **Stage 3**: Data Splitting - Apply Combinatorial Purged Cross-Validation (CPCV)
- **Stage 4**: Feature Engineering - Derive hand-crafted features from LOB snapshots
- **Stage 5**: LOB Standardization - Normalize LOB features
- **Stage 6**: Materialize Splits - Create split_X_input collections with 18 projected features
- **Stage 7**: Select Feature Transformations - Choose optimal transformations (16 features)
- **Stage 8**: Apply Feature Transformations - Apply transforms, rename output → input
- **Stage 9**: Stylized Facts Testing - Validate transformed features
- **Stage 10**: Select EWMA Half-Lives - Choose standardization parameters
- **Stage 11**: Apply EWMA Standardization - Standardize features, rename output → input
- **Stage 12**: Stylized Facts Testing - Validate standardized features
- **Stage 13**: Null Value Filtering - Remove documents with null/NaN/Inf values
- **Stage 14**: VQ-VAE Training - Train representation model

### Execution Pattern:
- **Swap Pattern**: Stages 2-5 use cyclic input/output pattern with collection swaps
- **Split Processing**: Stages 6+ work with individual split collections (split_X_input/output)
- **Analysis Stages**: Stages 7, 9, 12 are analysis stages that don't stop pipeline on failure

---

## 2. NULL FILTERING IMPLEMENTATION (Stage 13)

**Location**: `/home/user/drl-lob/scripts/12_filter_nulls.py`

### How Null Filtering Works:
1. **Detection Function**: Uses UDF `has_nulls_in_features()` to check each feature array
   - Detects None/null values
   - Detects NaN values
   - Detects Inf values

2. **Processing Strategy**:
   - Loads data in hourly batches (temporal ordering maintained)
   - Filters documents with null values in features array
   - Writes clean data to temporary collection (_output suffix)
   - Atomically replaces original collection

3. **Collection Handling**:
   - Input: split_X_input collections
   - Output: split_X_input collections (after replacement)
   - Temporary: split_X_output (intermediate storage)
   - Uses MongoDB drop and rename for atomic swap

4. **Statistics Tracking**:
   - Tracks documents removed by role (train/validation/test/etc)
   - Calculates removal percentage per split
   - Provides aggregate statistics across all splits

### Code Structure:
```python
# Per-split hourly processing
for split_id in range(MAX_SPLITS):
    for hour in all_hours:
        hour_df = load_hour_batch(split_id, hour)
        clean_df = hour_df.filter(~has_nulls_udf(col("features")))
        write_to_temp_collection(clean_df)
    # Atomic swap: temp → original
```

---

## 3. DATA FLOW AND STORAGE

### Storage Architecture:
- **Database**: MongoDB (raw database)
- **Collections Pattern**: 
  - Input/output cycling for pipeline stages
  - Split-specific collections: split_X_input, split_X_output
  - Metadata: ingestion_log, fold_assignment metadata

### Data Flow Between Stages:

```
Stage 2 (Ingestion)
    ↓ [parquet files → MongoDB]
    output → input (rename)
    ↓
Stage 3 (Splitting)
    input → process → output
    ↓ [add split_roles struct]
    output → input (swap)
    ↓
Stage 4 (Features)
    input → process → output
    ↓ [add feature arrays]
    output → input (swap)
    ↓
Stage 5 (Standardization)
    input → process → output
    ↓ [normalize features]
    output → input (swap)
    ↓
Stage 6 (Materialize Splits)
    input → [extract per split]
    ↓
    split_0_input, split_1_input, ... split_N_input
    [Each collection: 18 projected features]
    ↓
Stages 7-12 (Feature Processing)
    split_X_input → process → split_X_output → rename → split_X_input
    (Repeat for each transformation/standardization stage)
    ↓
Stage 13 (Null Filtering)
    split_X_input → [remove nulls] → split_X_output → rename → split_X_input
```

### Document Structure:

**Before Stage 6 (Materialization)**:
```javascript
{
  _id: ObjectId,
  timestamp: Date,
  timestamp_str: String,
  bids: [[price, volume], ...],  // Original LOB data
  asks: [[price, volume], ...],
  // Features:
  microprice: number,
  volatility: number,
  depth_*: numbers,              // Depth features
  hist_logret_*: numbers,        // Historical returns
  fwd_logret_1: number,          // Target
  // Metadata:
  fold_id: number,
  fold_type: string,             // 'train', 'test', etc
  split_roles: {                 // Key data structure!
    "0": "train",                // Split 0 role
    "1": "validation",           // Split 1 role
    ...                          // For each split
  }
}
```

**After Stage 6 (Materialization into split_X_input)**:
```javascript
{
  _id: ObjectId,
  timestamp: Date,
  features: [18 numbers],        // Projected features only
  feature_names: [18 strings],   // Feature name mappings
  role: string,                  // Role for THIS split
  fold_id: number,
  fold_type: string
}
```

### Key Data Structures:

#### split_roles Structure:
- Nested Map/Struct: Maps split_id (as string) → role (string)
- Roles include: 'train', 'validation', 'test', 'purged', 'embargoed', 'train_warmup', 'train_test_embargo', 'test_horizon'
- **This is the bridge between CPCV splits and individual sample roles**

#### Feature Arrays:
- 22 features total after Stage 4
- 18 features after projection (excludes mid_price, log_return, variance_proxy, spread)
- 16 features transformed/standardized (further excludes volatility and fwd_logret_1)

---

## 4. MONGODB COLLECTIONS - SPLITS COLLECTION

### Splits Collection Details:

**Collection Naming Pattern**: `split_X_input`, `split_X_output` (X = 0 to N-1)

**Number of Splits**: Determined by CPCV formula
- Formula: C(n_folds, k_validation_folds)
- Example: C(10, 2) = 45 possible splits (with n_folds=10, k_validation_folds=2)

**Collection Cardinality**:
- Each split collection contains samples for ALL roles in that split
- Before filtering: ~all training data partitioned by role
- After null filtering: Only valid samples retained

**Key Fields per Document**:
```
_id              → ObjectId (preserved through pipeline)
timestamp        → ISO datetime (UTC naive)
features         → Array of floats [F1, F2, ..., F18]
feature_names    → Array of strings ["name1", "name2", ..., "name18"]
role             → String (train/validation/test/purged/etc)
fold_id          → Integer (which fold this sample belongs to)
fold_type        → String (train/test_horizon/test/etc)
```

**Important Properties**:
1. **Temporal Ordering**: Documents are sorted by timestamp within each collection
2. **Feature Consistency**: All documents in a split have same feature_names array
3. **Role Semantics**: Role indicates sample's purpose in THIS split's CPCV fold
4. **Null Status**: After Stage 13, guaranteed no null/NaN/Inf values in features array

### Collection Evolution Through Pipeline:

| Stage | Collection | Features | Has split_roles | Contains |
|-------|------------|----------|-----------------|----------|
| 5 | input | 22 raw | Yes | All roles from all splits |
| 6 | split_X_input | 18 projected | No (role extracted) | Data for 1 specific split |
| 8 | split_X_input | 18 transformed | No | After transformation |
| 11 | split_X_input | 18 standardized | No | After EWMA standardization |
| 13 | split_X_input | 18 standardized | No | After null filtering (clean) |

---

## 5. S3 AND CLOUD STORAGE INTEGRATION

### Current Status: NO ACTIVE S3 INTEGRATION

**Evidence**:
1. **No S3 reads/writes in pipeline scripts**
2. **No AWS SDK calls in pipeline code**
3. **All data stored in local MongoDB**
4. **Parquet files stored locally (lob_data/ directory)**

**S3 Dependencies Present (but unused)**:
- boto3 (v1.40.46) in environment.yaml
- botocore (v1.40.46)
- s3transfer (v0.14.0)
- requests-auth-aws-sigv4

**Likely Intended For**:
- Future cloud deployment on AWS (references to g4dn.xlarge GPU instances in VQ-VAE config)
- Potential model artifact storage to S3
- Cloud-based training/hyperparameter search

**Data Collection Source**:
- BitMEX LOB WebSocket (wss://www.bitmex.com/realtime)
- Collects Bitcoin/USD order book snapshots every 30 seconds
- Stores directly to local Parquet files (lob_data/ directory)

### Storage Locations:

| Data Type | Location | Format |
|-----------|----------|--------|
| Raw LOB | `lob_data/*.parquet` | Parquet files (local filesystem) |
| Processed Data | MongoDB collections | BSON documents |
| Metadata | `artifacts/fold_assignment/reproducibility.yaml` | YAML |
| Feature Transforms | MLflow tracking | MLflow backend |
| Models | TBD | (Not yet implemented) |

---

## 6. DATA PASSING BETWEEN STAGES

### Mechanism 1: MongoDB Collections (Primary)
```
Stage N
  ├─ Read from: input collection
  ├─ Process: Spark DataFrames
  └─ Write to: output collection
        ↓
Pipeline Manager
  ├─ Drop: input collection
  ├─ Rename: output → input
  └─ Result: input ready for Stage N+1
        ↓
Stage N+1
  └─ Read from: input collection (output of previous stage)
```

### Mechanism 2: File-based (For Configuration)
- CPCV metadata saved to: `artifacts/fold_assignment/reproducibility.yaml`
- Feature transformation selections saved per MLflow runs
- Used for reproducibility and hyperparameter tracking

### Mechanism 3: Spark Broadcast Variables (For In-Memory Processing)
- Metadata broadcast to workers during stamping
- Folds information broadcast for role assignment
- Enables distributed processing of role assignment logic

### Data Integrity Safeguards:
1. **ObjectId Preservation**: Custom write function preserves MongoDB ObjectId format
2. **Temporal Ordering**: All writes are ordered, all reads are sorted by timestamp
3. **Atomic Swaps**: MongoDB rename operations are atomic (single transaction)
4. **Checkpoint Collections**: Temporary _output collections prevent data loss

---

## Summary Table: Pipeline Architecture

| Component | Type | Details |
|-----------|------|---------|
| **Orchestrator** | Python Script | run_pipeline.py (stages 2-14) |
| **Storage** | MongoDB | raw database with 45+ collections |
| **Data Format** | BSON/Spark | Documents with arrays and nested structures |
| **Feature Count** | Variable | 22 → 18 → 16 (progressive reduction) |
| **Splits** | CPCV | C(10,2) = 45 splits per run |
| **Collections** | Cyclic Pattern | input/output swapping between stages |
| **Null Filtering** | Stage 13 | UDF-based feature array validation |
| **Cloud Storage** | Not Active | S3 dependencies present but unused |
| **Data Source** | BitMEX WebSocket | Real-time LOB snapshots every 30s |

---

