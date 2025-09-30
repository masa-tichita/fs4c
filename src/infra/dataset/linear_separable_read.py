from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.dataset.linear_separable.contract import (
    LinearSeparableDataset,
    LinearSeparableDatasetConfig,
)


@dataclass(frozen=True)
class LinearSeparableDatasetReader:
    config: LinearSeparableDatasetConfig

    def generate_covariance_matrix(self) -> np.ndarray:
        """Toeplitz 型の共分散行列 Σ_ij = correlation^{|i-j|} を生成する。"""

        indices = np.arange(self.config.features, dtype=np.int32)
        distance_matrix = np.abs(indices[:, None] - indices[None, :])
        covariance = self.config.correlation**distance_matrix
        return covariance.astype(np.float64)

    def read(self) -> LinearSeparableDataset:
        """
        線形分離可能な合成データセットを生成する。

        Returns:
            LinearSeparableDataset: 特徴量・ラベル・真の係数を含むデータ構造。
        """
        # 乱数制御のインスタンスrngを作成
        rng = np.random.default_rng(self.config.seed)
        coefficients = self.config.build_coefficient_vector()
        covariance = self.generate_covariance_matrix()
        feature_matrix = rng.multivariate_normal(
            mean=np.zeros(self.config.features, dtype=np.float64),
            cov=covariance,
            size=self.config.size,
        )
        noise = (
            rng.normal(loc=0.0, scale=self.config.noise_std, size=self.config.size)
            if self.config.noise_std > 0.0
            else np.zeros(self.config.size, dtype=np.float64)
        )
        decision_values = feature_matrix @ coefficients + noise
        labels = np.where(decision_values >= 0.0, 1.0, -1.0).astype(np.float64)

        return LinearSeparableDataset(
            features=feature_matrix,
            labels=labels,
            coefficients=coefficients,
        )
