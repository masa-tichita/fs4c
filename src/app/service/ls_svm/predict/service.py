from __future__ import annotations

import numpy as np

from app.service.ls_svm.common.kernel import resolve_kernel_function
from app.service.ls_svm.predict.args import PredictLSSVMArgs
from app.service.ls_svm.predict.result import PredictLSSVMResult


class PredictLSSVMService:
    """LS-SVM モデルによる推論サービス。"""

    def execute(self, args: PredictLSSVMArgs) -> PredictLSSVMResult:
        feature_matrix = np.asarray(args.features, dtype=np.float64)
        kernel_fn = resolve_kernel_function(args.model.hyperparameters)
        gram_matrix = kernel_fn(feature_matrix, args.model.support_vectors)
        weights = args.model.alphas * args.model.support_labels
        decision_values = gram_matrix @ weights + args.model.bias
        predictions = np.where(decision_values >= 0.0, 1.0, -1.0)
        return PredictLSSVMResult(
            predictions=predictions,
            decision_values=decision_values,
        )
