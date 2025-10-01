from __future__ import annotations

import sys
from collections.abc import Sequence
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

DEFAULT_OUTPUT = Path("artifacts/ls_svm_visualization.png")


def parse_cli_args(
    argv: Sequence[str],
) -> tuple[
    LinearSeparableDatasetConfig,
    LSSVMHyperParameters,
    Path,
]:
    params: dict[str, str] = {}
    if argv:
        if len(argv) % 2 != 0:
            msg = "Arguments must be provided as --key value pairs."
            raise ValueError(msg)
        for flag, value in zip(argv[::2], argv[1::2]):
            if not flag.startswith("--"):
                msg = f"Invalid flag format: {flag}"
                raise ValueError(msg)
            key = flag[2:].replace("-", "_")
            params[key] = value

    dataset_params: dict[str, str] = {}
    hyper_params: dict[str, str] = {}
    output = Path(params.get("output", DEFAULT_OUTPUT))

    for key, value in params.items():
        if key in {
            "size",
            "features",
            "informative",
            "correlation",
            "noise_std",
            "snr",
            "seed",
            "train_ratio",
        }:
            dataset_params[key] = value
        elif key in {"gamma", "kernel", "rbf_sigma"}:
            hyper_params[key] = value

    if "features" not in dataset_params:
        dataset_params["features"] = "2"

    dataset_config = LinearSeparableDatasetConfig.model_validate(dataset_params)

    if "kernel" not in hyper_params:
        hyper_params["kernel"] = "linear"
    if hyper_params.get("kernel") == "linear" and "gamma" not in hyper_params:
        hyper_params["gamma"] = "10.0"

    hyperparameters = LSSVMHyperParameters.model_validate(hyper_params)

    return dataset_config, hyperparameters, output


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
    dataset_config, hyperparameters, output = parse_cli_args(sys.argv[1:])
    run_visualization(dataset_config, hyperparameters, output)


if __name__ == "__main__":
    main()
