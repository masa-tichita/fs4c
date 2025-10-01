from __future__ import annotations

import numpy as np

from app.ls_svm.usecase import PredictLSSVMUseCase, TrainLSSVMUseCase
from app.service.ls_svm.predict.args import PredictLSSVMArgs
from app.service.ls_svm.train.args import TrainLSSVMArgs
from domain.dataset.linear_separable.contract import LinearSeparableDatasetConfig
from domain.model.ls_svm.contract import LSSVMHyperParameters
from infra.dataset.linear_separable.read import LinearSeparableDatasetReader


def _generate_dataset(seed: int = 0):
    config = LinearSeparableDatasetConfig(size=12, noise_std=0.0, seed=seed)
    return LinearSeparableDatasetReader(config).read()


def test_train_and_predict_usecases_integration() -> None:
    dataset_split = _generate_dataset(seed=10)
    hyperparameters = LSSVMHyperParameters(gamma=5.0, kernel="linear")

    train_usecase = TrainLSSVMUseCase()
    predict_usecase = PredictLSSVMUseCase()

    train_result = train_usecase.execute(
        TrainLSSVMArgs(
            features=dataset_split.train.features,
            labels=dataset_split.train.labels,
            hyperparameters=hyperparameters,
        )
    )
    predict_result = predict_usecase.execute(
        PredictLSSVMArgs(
            model=train_result.model,
            features=dataset_split.test.features,
        )
    )

    assert np.array_equal(predict_result.predictions, dataset_split.test.labels)
