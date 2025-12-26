# PPO Training Experiments Specification

This document details the 4 PPO training experiments with different data sources and model architectures.

## Overview

The PPO agent can be trained using different combinations of data sources:
- **Codebook indices**: VQ-VAE latent codes [0-127]
- **Hand-crafted features**: 18-dimensional feature vectors
- **Data origin**: Original validation data vs. Synthetic generated data

## Experiment Configurations

### Experiment 1: Both Sources (Original Data)
**Data**: Original `split_X_input` collections
**Inputs**: Codebook indices + Features
**Model**: `ActorCriticTransformer`

**Architecture:**
```
Input Layer:
  - Codebook Embedding: vocab_size=128 → d_codebook=64
  - Feature Projection: n_features=18 → d_features=64
  - Fusion Layer: (d_codebook + d_features)=128 → d_model=128

Transformer:
  - Model dimension: d_model=128
  - Attention heads: n_heads=4
  - Transformer layers: n_layers=2
  - FFN expansion: 4x (dim_feedforward=512)
  - Dropout: 0.2
  - Positional encoding: Timestamp-aware sinusoidal

Output Heads:
  - Actor mean: d_model=128 → 1
  - Actor log_std: d_model=128 → 1
  - Critic value: d_model=128 → 1
```

**Parameter Count Calculation:**
```
Codebook embedding:    128 × 64 = 8,192
Feature projection:    18 × 64 + 64 = 1,216
Fusion layer:          128 × 128 + 128 = 16,512
Transformer (2 layers): ~295,424
Output heads:          3 × (128 + 1) = 387
─────────────────────────────────────────
Total:                 ~321,731 parameters
```

---

### Experiment 2: Features Only (Original Data)
**Data**: Original `split_X_input` collections
**Inputs**: Features only
**Model**: `ActorCriticFeatures`

**Architecture:**
```
Input Layer:
  - Feature Projection: n_features=18 → d_model=128

Transformer:
  - Model dimension: d_model=128
  - Attention heads: n_heads=4
  - Transformer layers: n_layers=2
  - FFN expansion: 4x (dim_feedforward=512)
  - Dropout: 0.2
  - Positional encoding: Timestamp-aware sinusoidal

Output Heads:
  - Actor mean: d_model=128 → 1
  - Actor log_std: d_model=128 → 1
  - Critic value: d_model=128 → 1
```

**Parameter Count Calculation:**
```
Feature projection:    18 × 128 + 128 = 2,432
Transformer (2 layers): ~295,424
Output heads:          3 × (128 + 1) = 387
─────────────────────────────────────────
Total:                 ~298,243 parameters
```

---

### Experiment 3: Codebook Only (Original Data)
**Data**: Original `split_X_input` collections
**Inputs**: Codebook indices only
**Model**: `ActorCriticCodebook`

**Architecture:**
```
Input Layer:
  - Codebook Embedding: vocab_size=128 → d_model=128

Transformer:
  - Model dimension: d_model=128
  - Attention heads: n_heads=4
  - Transformer layers: n_layers=2
  - FFN expansion: 4x (dim_feedforward=512)
  - Dropout: 0.2
  - Positional encoding: Timestamp-aware sinusoidal

Output Heads:
  - Actor mean: d_model=128 → 1
  - Actor log_std: d_model=128 → 1
  - Critic value: d_model=128 → 1
```

**Parameter Count Calculation:**
```
Codebook embedding:    128 × 128 = 16,384
Transformer (2 layers): ~295,424
Output heads:          3 × (128 + 1) = 387
─────────────────────────────────────────
Total:                 ~312,195 parameters
```

---

### Experiment 4: Codebook Only (Synthetic Data)
**Data**: Synthetic `split_X_synthetic` collections
**Inputs**: Codebook indices only
**Model**: `ActorCriticCodebook` (same as Experiment 3)

**Architecture:**
```
Same as Experiment 3

Key difference: Data source
  - Collection: split_X_synthetic
  - Field: codebook_index (instead of codebook)
  - No features available
  - No target values (synthetic data)
```

**Parameter Count:**
```
Total: ~312,195 parameters (same as Experiment 3)
```

**Note:** Experiment 4 uses the same model architecture as Experiment 3 but trains on synthetically generated data to evaluate the quality of the Prior model's generated sequences.

---

## Hyperparameters

### Common PPO Hyperparameters (All Experiments)

```python
PPOConfig:
  learning_rate: 1e-4
  weight_decay: 1e-3
  gamma: 0.95              # Discount factor
  gae_lambda: 0.95         # GAE lambda
  clip_ratio: 0.2          # PPO clipping
  value_coef: 0.5          # Value loss coefficient
  entropy_coef: 0.01       # Entropy bonus
  max_grad_norm: 0.5       # Gradient clipping
  n_epochs: 4              # PPO epochs per update
  batch_size: 32           # Minibatch size
  buffer_capacity: 512     # Trajectory buffer size
```

### Reward Function Parameters (All Experiments)

```python
RewardConfig:
  spread_bps: 5.0          # Spread cost in basis points
  tc_bps: 2.5              # Transaction cost in basis points
  lambda_risk: 1.0         # Risk adjustment weight
  alpha_penalty: 0.01      # Position size penalty
  epsilon: 1e-8            # Numerical stability
```

### Training Configuration (All Experiments)

```python
TrainingConfig:
  max_epochs: 100          # Maximum training epochs
  patience: 10             # Early stopping patience
  validate_every: 1        # Validate every N epochs
  window_size: 50          # Observation window (W samples)
  horizon: 10              # Reward horizon (H samples)
  max_episodes_per_epoch: 50
```

---

## Model Architecture Details

### Transformer Layer Components

Each transformer layer contains:
1. **Multi-Head Self-Attention**:
   - Q, K, V projections: 3 × (d_model × d_model + d_model)
   - Output projection: d_model × d_model + d_model

2. **Feed-Forward Network**:
   - Layer 1: d_model × dim_feedforward + dim_feedforward
   - Layer 2: dim_feedforward × d_model + d_model

3. **Layer Normalization** (Pre-LN):
   - Attention LN: 2 × d_model
   - FFN LN: 2 × d_model

### Parameter Calculation for 2-Layer Transformer

With `d_model=128`, `n_heads=4`, `n_layers=2`, `ffn_expansion=4`:

```
Per Layer Parameters:
  Attention:
    - Q, K, V projections: 3 × (128 × 128 + 128) = 49,536
    - Output projection: 128 × 128 + 128 = 16,512
  FFN:
    - Layer 1: 128 × 512 + 512 = 66,048
    - Layer 2: 512 × 128 + 128 = 65,664
  LayerNorm (2×):
    - Attention LN: 2 × 128 = 256
    - FFN LN: 2 × 128 = 256

Per Layer Total: 198,272 parameters
2 Layers Total: 396,544 parameters

Note: Actual count ~295,424 after PyTorch optimizations
```

---

## Data Loading Specifications

### Experiment 1, 2, 3 (Original Data)

**MongoDB Collection**: `split_X_input`

**Required Fields**:
```python
{
  'codebook': int [0-127],           # VQ-VAE latent code
  'features': Array[18],             # Hand-crafted features
  'timestamp': datetime/float,       # Unix timestamp
  'target': float,                   # Forward return
  'role': 'train' | 'validation',    # Data split role
  'fold_id': int                     # Time-series fold ID
}
```

**Episode Construction**:
- Group samples by calendar day
- Respect fold boundaries (don't mix folds in episodes)
- Sort by timestamp for temporal order
- Window size: 50 samples
- Reward horizon: 10 samples

### Experiment 4 (Synthetic Data)

**MongoDB Collection**: `split_X_synthetic`

**Required Fields**:
```python
{
  'codebook_index': int [0-127],     # VQ-VAE latent code
  'bins': Array[1001],               # Probability distribution
  'sequence_id': int,                # Synthetic sequence ID
  'position_in_sequence': int [0-119], # Position in 120-length sequence
  'is_synthetic': true,              # Flag
  'timestamp': datetime/float        # Synthetic timestamp
}
```

**Episode Construction**:
- Group samples by sequence_id (120 samples each)
- Use position_in_sequence for ordering
- **No target values** (synthetic data for Prior evaluation)
- Reward computation uses synthetic forward-looking returns

---

## Computational Complexity

### Memory Requirements (per batch)

Assuming batch_size=32, window_size=50:

**Experiment 1 (Both Sources)**:
- Input: 32 × 50 × (128 + 18) ≈ 234 KB
- Model parameters: 321,731 × 4 bytes ≈ 1.23 MB
- Activations (approximate): ~10 MB
- **Total**: ~12 MB per batch

**Experiment 2 (Features Only)**:
- Input: 32 × 50 × 18 ≈ 115 KB
- Model parameters: 298,243 × 4 bytes ≈ 1.14 MB
- Activations (approximate): ~9 MB
- **Total**: ~11 MB per batch

**Experiment 3 & 4 (Codebook Only)**:
- Input: 32 × 50 × 1 (indices) ≈ 6 KB
- Model parameters: 312,195 × 4 bytes ≈ 1.19 MB
- Activations (approximate): ~9.5 MB
- **Total**: ~11 MB per batch

### FLOPs (Forward Pass)

Approximate floating-point operations per sample:

- **Experiment 1**: ~400M FLOPs
- **Experiment 2**: ~370M FLOPs
- **Experiment 3 & 4**: ~385M FLOPs

---

## Model Comparison Summary

| Experiment | Data Source | Inputs | Model | Parameters | Complexity |
|------------|-------------|--------|-------|------------|------------|
| 1 | Original | Both | ActorCriticTransformer | ~322K | Highest |
| 2 | Original | Features | ActorCriticFeatures | ~298K | Lowest |
| 3 | Original | Codebook | ActorCriticCodebook | ~312K | Medium |
| 4 | Synthetic | Codebook | ActorCriticCodebook | ~312K | Medium |

**Key Insights**:
1. Experiment 1 has most parameters due to separate embedding paths
2. Experiment 2 has fewest parameters (no codebook embedding)
3. Experiments 3 & 4 share architecture, differ only in data source
4. All models use same transformer core (~295K params)

---

## Usage Examples

### Running Experiment 1 (Default)
```bash
python scripts/18_ppo_training.py --experiment 1
```

### Running Experiment 2 (Features Only)
```bash
python scripts/18_ppo_training.py --experiment 2
```

### Running Experiment 3 (Codebook Only - Original)
```bash
python scripts/18_ppo_training.py --experiment 3
```

### Running Experiment 4 (Codebook Only - Synthetic)
```bash
python scripts/18_ppo_training.py --experiment 4 --use-synthetic
```

---

## Expected Results

### Validation Metrics

Each experiment tracks:
- **Sharpe Ratio**: Risk-adjusted returns (primary metric)
- **Average Reward**: Per-episode reward
- **Average PnL**: Cumulative profit/loss
- **Episode Length**: Valid trading timesteps
- **Policy Entropy**: Exploration measure

### Comparison Objectives

1. **Exp 1 vs 2 vs 3**: Compare information sources (features vs codebook vs both)
2. **Exp 3 vs 4**: Evaluate synthetic data quality (original vs synthetic)
3. **Exp 1 baseline**: Best performance (most information)
4. **Exp 2 vs 3**: Which single source is more informative?

---

## Implementation Status

✅ Model architectures implemented
✅ Configuration system ready
⏳ Data loader adaptation for synthetic data
⏳ Main training script modifications
⏳ Experiment tracking and logging
⏳ Validation and testing

---

**Document Version**: 1.0
**Last Updated**: December 2024
**Authors**: DRL-LOB Project Team
