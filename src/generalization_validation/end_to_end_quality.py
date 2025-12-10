"""Experiment 3: End-to-End Synthetic Data Quality."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.logging import logger
from .data_loader import (
    load_validation_samples,
    load_vqvae_model,
    load_prior_model,
    decode_codes_batch
)
from .metrics import compute_mmd, compute_ks_tests, compute_correlation_distance
from .visualization import plot_umap_comparison, plot_correlation_matrices


class EndToEndValidator:
    """Validates end-to-end pipeline quality (Prior + VQ-VAE decoder)."""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        vqvae_model_dir: Path,
        prior_model_dir: Path,
        output_dir: Path,
        device: torch.device,
        seq_len: int = 120
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.vqvae_model_dir = vqvae_model_dir
        self.prior_model_dir = prior_model_dir
        self.output_dir = output_dir
        self.device = device
        self.seq_len = seq_len

        # Create output directory
        (self.output_dir / "experiment3_end_to_end").mkdir(parents=True, exist_ok=True)

    def validate_split(self, split_id: int, n_synthetic_samples: int = None) -> Dict:
        """
        Run end-to-end validation for one split.

        Args:
            split_id: Split identifier
            n_synthetic_samples: Number of synthetic samples to generate
                                 (default: match validation size)

        Returns:
            Dictionary with validation metrics
        """
        logger('', "INFO")
        logger(f'Validating end-to-end quality for split {split_id}...', "INFO")

        # Load validation data
        original_vectors, codebook_indices, _ = load_validation_samples(
            self.mongo_uri, self.db_name, split_id
        )

        n_val_samples = len(original_vectors)
        logger(f'  Validation samples: {n_val_samples:,}', "INFO")

        # Load models
        vqvae_path = self.vqvae_model_dir / f"split_{split_id}_model.pth"
        prior_path = self.prior_model_dir / f"split_{split_id}_prior.pth"

        if not vqvae_path.exists():
            raise FileNotFoundError(f"VQ-VAE model not found: {vqvae_path}")
        if not prior_path.exists():
            raise FileNotFoundError(f"Prior model not found: {prior_path}")

        vqvae_model = load_vqvae_model(vqvae_path, self.device)
        prior_model, prior_config = load_prior_model(prior_path, self.device)
        vocab_size = prior_config['vocab_size']

        # Get validation reconstructions (fair comparison)
        logger('  Decoding validation codes...', "INFO")
        val_reconstructed = decode_codes_batch(
            vqvae_model, codebook_indices, self.device, batch_size=512
        )

        # Generate synthetic data
        if n_synthetic_samples is None:
            n_synthetic_samples = n_val_samples

        logger(f'  Generating {n_synthetic_samples:,} synthetic samples...', "INFO")
        synthetic_vectors = self._generate_synthetic_data(
            prior_model, vqvae_model, n_synthetic_samples, vocab_size
        )

        logger(f'  Synthetic samples generated: {len(synthetic_vectors):,}', "INFO")

        # Compute metrics
        logger('  Computing MMD...', "INFO")
        mmd = compute_mmd(val_reconstructed, synthetic_vectors, kernel='rbf')
        logger(f'  MMD: {mmd:.6f}', "INFO")

        logger('  Running KS tests...', "INFO")
        ks_results = compute_ks_tests(val_reconstructed, synthetic_vectors)
        logger(f'  Mean KS statistic: {ks_results["mean_ks_statistic"]:.6f}', "INFO")
        logger(f'  Rejection rate: {ks_results["rejection_rate"]:.4f}', "INFO")

        logger('  Computing correlation distance...', "INFO")
        corr_results = compute_correlation_distance(val_reconstructed, synthetic_vectors)
        logger(f'  Correlation Frobenius norm: {corr_results["frobenius_norm"]:.6f}', "INFO")

        # Visualizations
        split_output_dir = self.output_dir / "experiment3_end_to_end" / f"split_{split_id}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # UMAP visualization
        logger('  Generating UMAP visualization...', "INFO")
        plot_umap_comparison(
            val_reconstructed, synthetic_vectors,
            title=f'End-to-End Quality - Split {split_id}',
            save_path=split_output_dir / f"umap_end_to_end_split_{split_id}.png",
            method='umap'
        )

        # Correlation matrices
        logger('  Plotting correlation matrices...', "INFO")
        plot_correlation_matrices(
            corr_results['corr_original'],
            corr_results['corr_synthetic'],
            save_path=split_output_dir / f"correlation_matrices_split_{split_id}.png",
            sample_features=100
        )

        # Marginal distribution comparisons (sample dimensions)
        logger('  Plotting marginal distributions...', "INFO")
        self._plot_marginals(
            val_reconstructed, synthetic_vectors,
            save_path=split_output_dir / f"marginals_split_{split_id}.png"
        )

        # Compile results
        results = {
            'split_id': split_id,
            'n_val_samples': n_val_samples,
            'n_syn_samples': len(synthetic_vectors),
            'mmd': float(mmd),
            'ks_mean_statistic': float(ks_results['mean_ks_statistic']),
            'ks_max_statistic': float(ks_results['max_ks_statistic']),
            'ks_rejection_rate': float(ks_results['rejection_rate']),
            'corr_frobenius': float(corr_results['frobenius_norm']),
            'corr_mean_abs_diff': float(corr_results['mean_absolute_diff']),
            'corr_max_abs_diff': float(corr_results['max_absolute_diff'])
        }

        logger(f'  ✓ Split {split_id} validation complete', "INFO")

        return results

    def _generate_synthetic_data(
        self,
        prior_model,
        vqvae_model,
        n_samples: int,
        vocab_size: int
    ) -> np.ndarray:
        """
        Generate synthetic data by sampling from Prior and decoding.

        Args:
            prior_model: Prior model
            vqvae_model: VQ-VAE model
            n_samples: Number of samples to generate
            vocab_size: Vocabulary size

        Returns:
            synthetic_vectors: (n_samples, 1001) array
        """
        prior_model.eval()
        vqvae_model.eval()

        synthetic_codes = []

        # Generate sequences
        n_sequences = (n_samples + self.seq_len - 1) // self.seq_len

        with torch.no_grad():
            for _ in range(n_sequences):
                # Generate sequence
                seq = prior_model.generate(
                    max_len=self.seq_len,
                    temperature=1.0,
                    device=self.device
                )

                synthetic_codes.extend(seq)

        # Trim to exact number of samples
        synthetic_codes = np.array(synthetic_codes[:n_samples], dtype=np.int64)

        # Decode
        synthetic_vectors = decode_codes_batch(
            vqvae_model, synthetic_codes, self.device, batch_size=512
        )

        return synthetic_vectors

    def _plot_marginals(
        self,
        val_data: np.ndarray,
        syn_data: np.ndarray,
        save_path: Path,
        n_features: int = 9
    ):
        """
        Plot marginal distributions for sample features.

        Args:
            val_data: Validation data
            syn_data: Synthetic data
            save_path: Path to save figure
            n_features: Number of features to plot
        """
        import matplotlib.pyplot as plt

        # Sample features evenly
        feature_indices = np.linspace(0, val_data.shape[1] - 1, n_features, dtype=int)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, feat_idx in enumerate(feature_indices):
            ax = axes[idx]

            ax.hist(val_data[:, feat_idx], bins=50, alpha=0.5, label='Validation',
                   density=True, color='blue')
            ax.hist(syn_data[:, feat_idx], bins=50, alpha=0.5, label='Synthetic',
                   density=True, color='red')

            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Feature {feat_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
