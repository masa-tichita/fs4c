from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

KernelLiteral = Literal["linear", "rbf"]


class LSSVMHyperParameters(BaseModel):
    """LS-SVM のハイパーパラメータ設定。"""

    gamma: float = Field(default=1.0, gt=0.0)
    kernel: KernelLiteral = Field(default="linear")
    rbf_sigma: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_kernel_params(self) -> "LSSVMHyperParameters":
        if self.kernel == "rbf" and self.rbf_sigma is None:
            msg = "rbf カーネルを使用する場合は rbf_sigma を指定してください。"
            raise ValueError(msg)
        return self


class LSSVMModel(BaseModel):
    """学習済み LS-SVM モデルの契約。"""

    bias: float
    alphas: NDArray[np.float64]
    support_vectors: NDArray[np.float64]
    support_labels: NDArray[np.float64]
    hyperparameters: LSSVMHyperParameters

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_shapes(self) -> "LSSVMModel":
        if self.support_vectors.shape[0] != self.support_labels.shape[0]:
            msg = "support_vectors と support_labels のサンプル数が一致していません。"
            raise ValueError(msg)
        if self.support_vectors.shape[0] != self.alphas.shape[0]:
            msg = "support_vectors のサンプル数と alphas の長さが一致していません。"
            raise ValueError(msg)
        return self
