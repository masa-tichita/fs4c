from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from domain.model.ls_svm.contract import LSSVMModel


@dataclass(frozen=True)
class PredictLSSVMArgs:
    model: LSSVMModel
    features: NDArray[np.float64]
