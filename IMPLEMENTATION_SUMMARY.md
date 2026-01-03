# Test Mode Implementation Summary

## Architecture

```
TEST MODE PIPELINE:
==================
Stage 7  (fit):   Fit transforms on split_0 → save test_mode/fitted_params.json
Stage 9  (apply): Load test_mode/fitted_params.json → apply to test_data
Stage 10 (fit):   Fit halflifes on split_0 → save test_mode/final_halflifes.json
                  Fit scalers on split_0 → save test_mode/scaler_states.json
Stage 12 (apply): Load test_mode/scaler_states.json → apply to test_data
```

## Implementation Status

### ✅ Stage 12 - COMPLETED
- Removed all fitting logic
- Only loads scaler states and applies
- Requires Stage 10 artifacts

### 🚧 Stage 10 - IN PROGRESS
Need to add:
1. `--mode train|test` argument
2. Test mode: process split_0 only
3. Save to `artifacts/ewma_halflife_selection/test_mode/`
4. Fit scalers and save to `artifacts/ewma_standardization/scaler_states/`

### 🚧 Stage 7 - PENDING
Need to add:
1. `--mode train|test` argument
2. Test mode: process split_0 only
3. Save to `artifacts/feature_transformation/test_mode/`

### 🚧 Stage 9 - PENDING
Need to modify:
1. Test mode: load from `test_mode/` instead of `split_0/`

## File Paths

```
artifacts/
├── feature_transformation/
│   ├── split_0/fitted_params.json          # Train mode
│   └── test_mode/fitted_params.json        # Test mode ← NEW
├── ewma_halflife_selection/
│   ├── aggregation/final_halflifes.json    # Train mode
│   └── test_mode/final_halflifes.json      # Test mode ← NEW
└── ewma_standardization/
    └── scaler_states/
        └── test_mode_scaler_states.json    # Test mode ← NEW (Stage 10 creates)
```

## Usage

```bash
# Test Mode Pipeline (in order):
python scripts/07_feature_transform.py --mode test --test-split 0
python scripts/08_apply_feature_transforms.py --mode test --test-split 0
python scripts/10_feature_scale.py --mode test --test-split 0
python scripts/11_apply_feature_standardization.py --mode test --test-split 0
```
