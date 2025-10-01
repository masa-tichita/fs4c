from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_noise_std_from_snr(
    coefficients: NDArray[np.float64],
    covariance: NDArray[np.float64],
    snr: float,
) -> float:
    """Calculate noise standard deviation to match a desired SNR."""

    if snr <= 0.0:
        msg = "snr must be positive."
        raise ValueError(msg)

    signal_variance = float(coefficients.T @ covariance @ coefficients)
    if signal_variance <= 0.0:
        msg = "signal variance must be positive to compute noise_std."
        raise ValueError(msg)

    return float(np.sqrt(signal_variance / snr))
