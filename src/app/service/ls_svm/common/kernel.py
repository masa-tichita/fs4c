from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from domain.model.ls_svm.contract import LSSVMHyperParameters


def resolve_kernel_function(
    hyperparameters: LSSVMHyperParameters,
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    if hyperparameters.kernel == "linear":

        def linear(
            lhs: NDArray[np.float64], rhs: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return lhs @ rhs.T

        return linear

    sigma = hyperparameters.rbf_sigma
    if sigma is None:
        msg = "rbf カーネルを使用する場合は rbf_sigma を指定してください。"
        raise ValueError(msg)
    sigma_value = float(sigma)
    sigma_sq = 2.0 * sigma_value**2

    def rbf(lhs: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        lhs_expanded = lhs[:, None, :]
        rhs_expanded = rhs[None, :, :]
        diff = lhs_expanded - rhs_expanded
        sq_dist = np.sum(diff * diff, axis=2)
        return np.exp(-sq_dist / sigma_sq)

    return rbf
