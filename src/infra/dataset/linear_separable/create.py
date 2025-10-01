from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from domain.dataset.linear_separable.contract import (
    LinearSeparableDataset,
    LinearSeparableDatasetConfig,
    LinearSeparableDatasetSplit,
)
from utils.statistics import compute_noise_std_from_snr


def _generate_covariance_matrix(
    features: int, correlation: float
) -> NDArray[np.float64]:
    indices = np.arange(features, dtype=np.int32)
    distance_matrix = np.abs(indices[:, None] - indices[None, :])
    covariance = correlation**distance_matrix
    return covariance.astype(np.float64)


def _determine_noise_std(
    coefficients: NDArray[np.float64],
    covariance: NDArray[np.float64],
    noise_std: float,
    snr: float | None,
) -> float:
    if noise_std > 0.0 or snr is None:
        return float(noise_std)
    return compute_noise_std_from_snr(coefficients, covariance, snr)


def create_linear_separable_split(
    config: LinearSeparableDatasetConfig,
    rng: np.random.Generator,
) -> LinearSeparableDatasetSplit:
    coefficients = config.build_coefficient_vector()
    covariance = _generate_covariance_matrix(config.features, config.correlation)
    noise_std = _determine_noise_std(
        coefficients=coefficients,
        covariance=covariance,
        noise_std=config.noise_std,
        snr=config.snr,
    )

    features = rng.multivariate_normal(
        mean=np.zeros(config.features, dtype=np.float64),
        cov=covariance,
        size=config.size,
    )
    noise = (
        rng.normal(loc=0.0, scale=noise_std, size=config.size)
        if noise_std > 0.0
        else np.zeros(config.size, dtype=np.float64)
    )
    decision_values = features @ coefficients + noise
    labels = np.where(decision_values >= 0.0, 1.0, -1.0).astype(np.float64)

    permutation = rng.permutation(config.size)
    features = features[permutation]
    labels = labels[permutation]

    train_size = max(1, int(round(config.size * config.train_ratio)))
    if train_size >= config.size:
        train_size = config.size - 1

    train_dataset = LinearSeparableDataset(
        features=features[:train_size],
        labels=labels[:train_size],
        coefficients=coefficients,
    )
    test_dataset = LinearSeparableDataset(
        features=features[train_size:],
        labels=labels[train_size:],
        coefficients=coefficients,
    )

    return LinearSeparableDatasetSplit(train=train_dataset, test=test_dataset)
