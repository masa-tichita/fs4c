from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.service.ls_svm.predict.args import PredictLSSVMArgs
from app.service.ls_svm.predict.result import PredictLSSVMResult
from app.service.ls_svm.predict.service import PredictLSSVMService
from app.service.ls_svm.train.args import TrainLSSVMArgs
from app.service.ls_svm.train.result import TrainLSSVMResult
from app.service.ls_svm.train.service import TrainLSSVMService
from utils.logging import log_fn


@dataclass
class TrainLSSVMUseCase:
    service: TrainLSSVMService = field(default_factory=TrainLSSVMService)

    @log_fn
    def execute(self, args: TrainLSSVMArgs) -> TrainLSSVMResult:
        return self.service.execute(args)


@dataclass
class PredictLSSVMUseCase:
    service: PredictLSSVMService = field(default_factory=PredictLSSVMService)

    @log_fn
    def execute(self, args: PredictLSSVMArgs) -> PredictLSSVMResult:
        return self.service.execute(args)

    def predict(self, args: PredictLSSVMArgs) -> NDArray[np.float64]:
        return self.execute(args).predictions

    def decision_function(self, args: PredictLSSVMArgs) -> NDArray[np.float64]:
        return self.execute(args).decision_values
