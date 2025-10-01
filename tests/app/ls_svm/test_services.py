from __future__ import annotations

import numpy as np
import pytest

from app.service.ls_svm.predict.args import PredictLSSVMArgs
from app.service.ls_svm.predict.service import PredictLSSVMService
from app.service.ls_svm.train.args import TrainLSSVMArgs
from app.service.ls_svm.train.service import TrainLSSVMService
from domain.dataset.linear_separable.contract import LinearSeparableDatasetConfig
from domain.model.ls_svm.contract import LSSVMHyperParameters
from infra.dataset.linear_separable.read import LinearSeparableDatasetReader


def _generate_dataset(
    size: int = 20,
    noise_std: float = 0.0,
    seed: int = 0,
):
    config = LinearSeparableDatasetConfig(size=size, noise_std=noise_std, seed=seed)
    return LinearSeparableDatasetReader(config).read()


def test_train_service_produces_model() -> None:
    dataset_split = _generate_dataset(size=15, seed=1)
    args = TrainLSSVMArgs(
        features=dataset_split.train.features,
        labels=dataset_split.train.labels,
        hyperparameters=LSSVMHyperParameters(gamma=10.0, kernel="linear"),
    )

    service = TrainLSSVMService()
    result = service.execute(args)

    inference_args = PredictLSSVMArgs(
        model=result.model,
        features=dataset_split.train.features,
    )
    predictions = PredictLSSVMService().execute(inference_args).predictions

    assert np.array_equal(predictions, dataset_split.train.labels)


def test_predict_service_returns_decision_values() -> None:
    dataset_split = _generate_dataset(size=12, seed=2)
    train_args = TrainLSSVMArgs(
        features=dataset_split.train.features,
        labels=dataset_split.train.labels,
        hyperparameters=LSSVMHyperParameters(gamma=5.0, kernel="rbf", rbf_sigma=0.5),
    )
    model = TrainLSSVMService().execute(train_args).model

    predict_args = PredictLSSVMArgs(model=model, features=dataset_split.test.features)
    result = PredictLSSVMService().execute(predict_args)

    assert result.decision_values.shape == (dataset_split.test.features.shape[0],)
    assert np.all(np.isfinite(result.decision_values))


def test_train_service_fallback_when_gurobi_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_split = _generate_dataset(size=8, seed=4)
    args = TrainLSSVMArgs(
        features=dataset_split.train.features,
        labels=dataset_split.train.labels,
        hyperparameters=LSSVMHyperParameters(gamma=1.0, kernel="linear"),
    )

    service = TrainLSSVMService()

    fallback_used = {"called": False}

    def fake_solve_with_gurobi(self, matrix, rhs):  # type: ignore[override]
        fallback_used["called"] = True
        raise RuntimeError("force fallback")

    monkeypatch.setattr(
        TrainLSSVMService,
        "_solve_with_gurobi",
        fake_solve_with_gurobi,
    )

    result = service.execute(args)
    predictions = (
        PredictLSSVMService()
        .execute(
            PredictLSSVMArgs(
                model=result.model,
                features=dataset_split.test.features,
            )
        )
        .predictions
    )

    assert np.array_equal(predictions, dataset_split.test.labels)
    assert fallback_used["called"] is True
