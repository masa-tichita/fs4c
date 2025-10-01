from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.dataset.linear_separable.contract import (
    LinearSeparableDatasetConfig,
    LinearSeparableDatasetSplit,
)
from infra.dataset.linear_separable.create import create_linear_separable_split


@dataclass(frozen=True)
class LinearSeparableDatasetReader:
    config: LinearSeparableDatasetConfig

    def read(self) -> LinearSeparableDatasetSplit:
        rng = np.random.default_rng(self.config.seed)
        return create_linear_separable_split(self.config, rng)
