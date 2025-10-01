from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LinearSeparableDatasetConfig(BaseModel):
    """合成データ生成のパラメータ設定。"""

    size: int = Field(default=100, ge=1)
    features: int = Field(default=25, ge=1)
    informative: Optional[int] = Field(default=None, ge=1)
    correlation: float = Field(default=0.35, ge=0.0, lt=1.0)
    noise_std: float = Field(default=0.0, ge=0.0)
    snr: Optional[float] = Field(default=None, gt=0.0)
    seed: Optional[int] = None
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def check_dimensions(self) -> "LinearSeparableDatasetConfig":
        if self.informative is None:
            return self
        if self.informative > self.features:
            msg = "informative must not exceed features."
            raise ValueError(msg)
        max_required_index = 2 + 3 * (self.informative - 1)
        if max_required_index >= self.features:
            msg = (
                "features must be greater than the largest informative index derived from "
                "the (0, 0, 1) pattern. Increase features or reduce informative."
            )
            raise ValueError(msg)
        return self

    def build_coefficient_vector(self) -> np.ndarray:
        """真の係数ベクトル a* (真に選択される特徴量を表現する0-1変数の集合)を生成する。"""
        if self.informative is None:
            return np.ones(self.features, dtype=np.float64)

        coefficients = np.zeros(self.features, dtype=np.float64)
        informative_indices = 2 + 3 * np.arange(self.informative)
        coefficients[informative_indices] = 1.0
        return coefficients


class LinearSeparableDataset(BaseModel):
    """生成された特徴量とラベルのセット。"""

    features: np.ndarray
    labels: np.ndarray
    coefficients: np.ndarray

    # pydanticで許容されていない型を許可する設定
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_shapes(self) -> "LinearSeparableDataset":
        if self.features.shape[0] != self.labels.shape[0]:
            msg = "features と labels のサンプル数が一致していません。"
            raise ValueError(msg)
        if self.features.shape[1] != self.coefficients.shape[0]:
            msg = "features の列数と coefficients の次元が一致していません。"
            raise ValueError(msg)
        return self


class LinearSeparableDatasetSplit(BaseModel):
    """Train/Test 分割済みデータセット。"""

    train: LinearSeparableDataset
    test: LinearSeparableDataset

    model_config = ConfigDict(arbitrary_types_allowed=True)
