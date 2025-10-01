from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ls_svm.usecase import PredictLSSVMUseCase, TrainLSSVMUseCase
from app.service.ls_svm.predict.args import PredictLSSVMArgs
from app.service.ls_svm.train.args import TrainLSSVMArgs
from domain.dataset.linear_separable.contract import LinearSeparableDatasetConfig
from domain.model.ls_svm.contract import LSSVMHyperParameters
from infra.dataset.linear_separable.read import LinearSeparableDatasetReader
from utils.logging import log_fn, setup_logger

DATASET_CONFIG = LinearSeparableDatasetConfig(
    size=100,
    features=25,
    snr=1.0,
    seed=42,
    train_ratio=0.7,
)

HYPER_PARAMETERS = LSSVMHyperParameters(gamma=10.0, kernel="linear")

OUTPUT_PATH = Path("artifacts/ls_svm_visualization.png")


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@log_fn
def run_visualization(
    dataset_config: LinearSeparableDatasetConfig,
    hyperparameters: LSSVMHyperParameters,
    output: Path,
) -> float:
    dataset_split = LinearSeparableDatasetReader(dataset_config).read()

    train_result = TrainLSSVMUseCase().execute(
        TrainLSSVMArgs(
            features=dataset_split.train.features,
            labels=dataset_split.train.labels,
            hyperparameters=hyperparameters,
        )
    )

    predict_result = PredictLSSVMUseCase().execute(
        PredictLSSVMArgs(
            model=train_result.model,
            features=dataset_split.test.features,
        )
    )

    ensure_output_directory(output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_kwargs = {"s": 40, "alpha": 0.8}

    axes[0].set_title("Ground Truth (Train)")
    axes[0].scatter(
        dataset_split.train.features[:, 0],
        dataset_split.train.features[:, 1],
        c=dataset_split.train.labels,
        cmap="bwr",
        **scatter_kwargs,
    )
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    axes[1].set_title("Predicted Labels (Test)")
    axes[1].scatter(
        dataset_split.test.features[:, 0],
        dataset_split.test.features[:, 1],
        c=predict_result.predictions,
        cmap="bwr",
        **scatter_kwargs,
    )
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)

    accuracy = float(np.mean(predict_result.predictions == dataset_split.test.labels))
    logger.info("Saved visualization to {} (accuracy={:.3f})", output, accuracy)
    return accuracy


def main() -> None:
    setup_logger()
    run_visualization(DATASET_CONFIG, HYPER_PARAMETERS, OUTPUT_PATH)


if __name__ == "__main__":
    main()
