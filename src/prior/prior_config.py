"""
Prior Model Configuration

Hyperparameter grids and training configurations for latent prior learning.
"""

# =================================================================================================
# Hyperparameter Grid for Search
# =================================================================================================
# Budget-optimized: 18 configurations (aligned with VQ-VAE architecture K=128, D=64)
#
# Expected runtime (assuming ~5s per epoch with prefetching):
# - Worst case (75 epochs per split): 18 configs × 28 splits × 75 epochs × 5s = 189,000s = 52.5 hours = $37.28
# - Expected case (early stopping ~15 epochs): 18 configs × 28 splits × 15 epochs × 5s = 37,800s = 10.5 hours = $7.46
# - Per-split estimate: 18 configs × 15 epochs × 5s = 1,350s = 22.5 minutes
#
# Note: Prior model is much faster than VQVAE (smaller model, integer sequences vs LOB bins)
# Embedding dimensions [64, 96, 128] align with VQ-VAE's D=64 (match, 1.5×, 2×)
# =================================================================================================

PRIOR_HYPERPARAM_GRID = {
    # Architecture parameters
    # Note: VQ-VAE uses K=128 (codebook size) and D=64 (embedding dimension)
    # Prior embedding_dim should align with or relate to VQ-VAE's D=64
    'embedding_dim': [64, 96, 128],      # Code embedding size: match D, 1.5×D, 2×D
    'n_layers': [6, 8, 10],              # Causal CNN depth (receptive field)
    'n_channels': [64, 80],              # Causal CNN width (capacity)
    'kernel_size': [2],                  # Standard for causal convolutions

    # Training parameters
    'learning_rate': [1e-3],             # Adam learning rate
    'dropout': [0.15],                   # Regularization
}
# Total: 3 × 3 × 2 = 18 configurations
# Parameter range: aligned with VQ-VAE architecture (K=128, D=64)

# =================================================================================================
# Training Configuration
# =================================================================================================

PRIOR_TRAINING_CONFIG = {
    'max_epochs': 75,
    'patience': 5,                       # Early stopping patience
    'min_delta': 0.001,                  # Minimum improvement threshold
    'hours_per_accumulation': 100,      # Match VQ-VAE for GPU efficiency
    'sequence_batch_size': 32,           # Number of sequences per GPU batch
    'seq_len': 120,                      # 1 hour = ~120 LOB snapshots
    'grad_clip_norm': 1.0,
    'weight_decay': 1e-5,
}

# =================================================================================================
# Generation Configuration
# =================================================================================================

GENERATION_CONFIG = {
    'sequences_per_split': 100,          # Number of synthetic sequences per split
    'seq_len': 120,                      # 1 hour sequences
    'temperature': 1.0,                  # Sampling temperature (match training)
    'generation_batch_size': 32,         # Generate 32 sequences at once (GPU)
}