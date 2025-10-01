from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictLSSVMResult:
    predictions: NDArray[np.float64]
    decision_values: NDArray[np.float64]
