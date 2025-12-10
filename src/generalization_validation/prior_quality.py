"""Experiment 2: Prior Model Quality Assessment."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.logging import logger
from .data_loader import load_validation_samples, load_prior_model, organize_codes_into_sequences
from .metrics import (
    compute_code_frequency,
    compute_transition_matrix,
    extract_ngrams,
    compare_ngrams,
    jensen_shannon_divergence
)
from .visualization import (
    plot_code_frequency,
    plot_transition_matrix,
    plot_ngram_comparison,
    plot_umap_comparison
)


class PriorQualityValidator:
    """Validates Prior model's ability to generate realistic code sequences."""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        prior_model_dir: Path,
        output_dir: Path,
        device: torch.device,
        seq_len: int = 120
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.prior_model_dir = prior_model_dir
        self.output_dir = output_dir
        self.device = device
        self.seq_len = seq_len

        # Create output directory
        (self.output_dir / "experiment2_prior_quality").mkdir(parents=True, exist_ok=True)

    def validate_split(self, split_id: int) -> Dict:
        """
        Run Prior quality validation for one split.

        Args:
            split_id: Split identifier

        Returns:
            Dictionary with validation metrics
        """
        logger('', "INFO")
        logger(f'Validating Prior model for split {split_id}...', "INFO")

        # Load validation codes
        _, codebook_indices, _ = load_validation_samples(
            self.mongo_uri, self.db_name, split_id
        )

        logger(f'  Validation codes: {len(codebook_indices):,}', "INFO")

        # Organize into sequences
        val_sequences = organize_codes_into_sequences(
            codebook_indices, self.seq_len, stride=self.seq_len
        )

        n_sequences = len(val_sequences)
        logger(f'  Validation sequences: {n_sequences}', "INFO")

        # Load Prior model
        model_path = self.prior_model_dir / f"split_{split_id}_prior.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Prior model not found: {model_path}")

        prior_model, prior_config = load_prior_model(model_path, self.device)
        vocab_size = prior_config['vocab_size']

        # Generate synthetic sequences
        logger('  Generating synthetic sequences...', "INFO")
        syn_sequences = self._generate_sequences(prior_model, n_sequences, vocab_size)

        # 1. Code frequency distributions
        logger('  Analyzing code frequencies...', "INFO")
        freq_val = compute_code_frequency(val_sequences, vocab_size)
        freq_syn = compute_code_frequency(syn_sequences, vocab_size)

        # JS divergence
        js_div = jensen_shannon_divergence(freq_val, freq_syn)
        logger(f'  JS Divergence (frequencies): {js_div:.6f}', "INFO")

        # Frequency correlation
        freq_corr = np.corrcoef(freq_val, freq_syn)[0, 1]
        logger(f'  Frequency correlation: {freq_corr:.6f}', "INFO")

        # 2. Transition probabilities
        logger('  Computing transition matrices...', "INFO")
        trans_val = compute_transition_matrix(val_sequences, vocab_size)
        trans_syn = compute_transition_matrix(syn_sequences, vocab_size)

        # Frobenius norm
        trans_frobenius = np.linalg.norm(trans_val - trans_syn, ord='fro')
        logger(f'  Transition matrix Frobenius norm: {trans_frobenius:.6f}', "INFO")

        # 3. N-gram statistics
        logger('  Extracting n-grams...', "INFO")

        # Bigrams
        bigrams_val = extract_ngrams(val_sequences, 2)
        bigrams_syn = extract_ngrams(syn_sequences, 2)
        bigram_comparison = compare_ngrams(bigrams_val, bigrams_syn, top_k=20)

        logger(f'  Bigram overlap ratio: {bigram_comparison["overlap_ratio"]:.4f}', "INFO")
        logger(f'  Bigram frequency correlation: {bigram_comparison["frequency_correlation"]:.6f}', "INFO")

        # Trigrams
        trigrams_val = extract_ngrams(val_sequences, 3)
        trigrams_syn = extract_ngrams(syn_sequences, 3)
        trigram_comparison = compare_ngrams(trigrams_val, trigrams_syn, top_k=20)

        logger(f'  Trigram overlap ratio: {trigram_comparison["overlap_ratio"]:.4f}', "INFO")

        # Visualizations
        split_output_dir = self.output_dir / "experiment2_prior_quality" / f"split_{split_id}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Code frequency plot
        logger('  Plotting code frequencies...', "INFO")
        plot_code_frequency(
            freq_val, freq_syn,
            save_path=split_output_dir / f"code_frequency_split_{split_id}.png"
        )

        # Transition matrix plot
        logger('  Plotting transition matrices...', "INFO")
        plot_transition_matrix(
            trans_val, trans_syn,
            save_path=split_output_dir / f"transition_matrix_split_{split_id}.png"
        )

        # Bigram comparison plot
        logger('  Plotting bigrams...', "INFO")
        plot_ngram_comparison(
            bigram_comparison['top_val_ngrams'],
            bigram_comparison['top_syn_ngrams'],
            n=2,
            save_path=split_output_dir / f"bigrams_split_{split_id}.png"
        )

        # Trigram comparison plot
        logger('  Plotting trigrams...', "INFO")
        plot_ngram_comparison(
            trigram_comparison['top_val_ngrams'],
            trigram_comparison['top_syn_ngrams'],
            n=3,
            save_path=split_output_dir / f"trigrams_split_{split_id}.png"
        )

        # UMAP on sequences (treat sequences as high-dimensional points)
        logger('  Generating sequence UMAP...', "INFO")
        plot_umap_comparison(
            val_sequences.astype(np.float32),
            syn_sequences.astype(np.float32),
            title=f'Prior Sequences - Split {split_id}',
            save_path=split_output_dir / f"umap_sequences_split_{split_id}.png",
            method='umap'
        )

        # Compile results
        results = {
            'split_id': split_id,
            'n_val_sequences': n_sequences,
            'n_syn_sequences': len(syn_sequences),
            'seq_len': self.seq_len,
            'vocab_size': vocab_size,
            'js_divergence_freq': float(js_div),
            'frequency_correlation': float(freq_corr),
            'transition_frobenius': float(trans_frobenius),
            'bigram_overlap_ratio': float(bigram_comparison['overlap_ratio']),
            'bigram_freq_correlation': float(bigram_comparison['frequency_correlation']),
            'trigram_overlap_ratio': float(trigram_comparison['overlap_ratio']),
            'trigram_freq_correlation': float(trigram_comparison['frequency_correlation'])
        }

        logger(f'  ✓ Split {split_id} validation complete', "INFO")

        return results

    def _generate_sequences(
        self,
        model,
        n_sequences: int,
        vocab_size: int
    ) -> np.ndarray:
        """
        Generate synthetic sequences using Prior model.

        Args:
            model: Prior model
            n_sequences: Number of sequences to generate
            vocab_size: Vocabulary size

        Returns:
            sequences: (n_sequences, seq_len) array
        """
        model.eval()
        sequences = []

        with torch.no_grad():
            for _ in range(n_sequences):
                # Generate sequence
                seq = model.generate(
                    max_len=self.seq_len,
                    temperature=1.0,
                    device=self.device
                )

                # Ensure correct length
                if len(seq) >= self.seq_len:
                    seq = seq[:self.seq_len]
                else:
                    # Pad if needed (shouldn't happen with proper generation)
                    seq = np.pad(seq, (0, self.seq_len - len(seq)), mode='edge')

                sequences.append(seq)

        sequences = np.array(sequences, dtype=np.int64)

        return sequences
