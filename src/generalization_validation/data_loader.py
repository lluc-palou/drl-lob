"""Data loading utilities for generalization validation."""

import torch
import numpy as np
from pymongo import MongoClient
from pathlib import Path
from typing import Dict, List, Tuple
from src.utils.logging import logger


def load_validation_samples(
    mongo_uri: str,
    db_name: str,
    split_id: int,
    collection_suffix: str = "_input"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load validation samples for a split.

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Database name
        split_id: Split identifier
        collection_suffix: Collection suffix (default: "_input")

    Returns:
        original_vectors: (N, 1001) array of original LOB vectors
        codebook_indices: (N,) array of VQ-VAE codes
        timestamps: (N,) array of timestamps
    """
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    collection_name = f"split_{split_id}{collection_suffix}"

    collection = db[collection_name]

    # Query validation samples (both validation folds)
    cursor = collection.find(
        {'role': 'validation'},
        {'bins': 1, 'codebook': 1, 'timestamp': 1}
    ).sort('timestamp', 1)

    original_vectors = []
    codebook_indices = []
    timestamps = []

    for doc in cursor:
        original_vectors.append(doc['bins'])
        codebook_indices.append(doc['codebook'])

        # Handle timestamp
        ts = doc['timestamp']
        if hasattr(ts, 'timestamp'):
            timestamps.append(ts.timestamp())
        else:
            timestamps.append(float(ts))

    client.close()

    logger(f'Loaded {len(original_vectors)} validation samples for split {split_id}', "INFO")

    return (
        np.array(original_vectors, dtype=np.float32),
        np.array(codebook_indices, dtype=np.int64),
        np.array(timestamps, dtype=np.float64)
    )


def load_vqvae_model(model_path: Path, device: torch.device):
    """
    Load VQ-VAE model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        VQ-VAE model
    """
    from src.vqvae_representation.model import VQVAEModel

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = VQVAEModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger(f'Loaded VQ-VAE model from {model_path}', "INFO")

    return model


def load_prior_model(model_path: Path, device: torch.device):
    """
    Load Prior model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Prior model and config
    """
    from src.prior.prior_model import TransformerPrior

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Create model with config
    model = TransformerPrior(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config.get('max_seq_len', 256)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger(f'Loaded Prior model from {model_path}', "INFO")

    return model, config


def organize_codes_into_sequences(
    codes: np.ndarray,
    seq_len: int,
    stride: int = None
) -> np.ndarray:
    """
    Organize discrete codes into sequences.

    Args:
        codes: (N,) array of code indices
        seq_len: Sequence length
        stride: Stride for creating sequences (default: seq_len for non-overlapping)

    Returns:
        sequences: (num_sequences, seq_len) array of code sequences
    """
    if stride is None:
        stride = seq_len

    n_codes = len(codes)
    sequences = []

    for start_idx in range(0, n_codes - seq_len + 1, stride):
        seq = codes[start_idx:start_idx + seq_len]
        sequences.append(seq)

    sequences = np.array(sequences, dtype=np.int64)

    logger(f'Created {len(sequences)} sequences of length {seq_len}', "INFO")

    return sequences


def decode_codes_batch(
    model,
    codes: np.ndarray,
    device: torch.device,
    batch_size: int = 512
) -> np.ndarray:
    """
    Decode codebook indices to vectors in batches.

    Args:
        model: VQ-VAE model
        codes: (N,) array of code indices
        device: Device
        batch_size: Batch size for decoding

    Returns:
        decoded: (N, 1001) array of decoded vectors
    """
    model.eval()
    decoded_vectors = []

    n_codes = len(codes)

    with torch.no_grad():
        for start_idx in range(0, n_codes, batch_size):
            end_idx = min(start_idx + batch_size, n_codes)
            batch_codes = codes[start_idx:end_idx]

            # Convert to tensor
            batch_codes_tensor = torch.tensor(batch_codes, dtype=torch.long, device=device)

            # Get codebook embeddings
            z_q = model.vq.embedding(batch_codes_tensor)

            # Decode
            reconstructed = model.decoder(z_q)

            decoded_vectors.append(reconstructed.cpu().numpy())

    decoded = np.concatenate(decoded_vectors, axis=0)

    return decoded
