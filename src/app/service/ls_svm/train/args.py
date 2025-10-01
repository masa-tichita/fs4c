from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from domain.model.ls_svm.contract import LSSVMHyperParameters


@dataclass(frozen=True)
class TrainLSSVMArgs:
    features: NDArray[np.float64]
    labels: NDArray[np.float64]
    hyperparameters: LSSVMHyperParameters
