from __future__ import annotations

import numpy as np
import pytest

from domain.dataset.linear_separable.contract import LinearSeparableDatasetConfig
from infra.dataset.linear_separable.read import LinearSeparableDatasetReader


class TestLinearSeparableDatasetReader:
    @pytest.fixture(autouse=True)
    def _setup_reader(self) -> None:
        self.reader = LinearSeparableDatasetReader(
            LinearSeparableDatasetConfig(size=7, features=4, seed=123)
        )

    def test_reader_returns_expected_shapes(self) -> None:
        dataset_split = self.reader.read()

        assert dataset_split.train.features.shape[1] == 4
        assert dataset_split.test.features.shape[1] == 4
        assert (
            dataset_split.train.features.shape[0] + dataset_split.test.features.shape[0]
            == 7
        )
        assert dataset_split.train.coefficients.shape == (4,)
        unique_labels = np.unique(
            np.concatenate([dataset_split.train.labels, dataset_split.test.labels])
        )
        assert set(unique_labels).issubset({-1.0, 1.0})

    def test_reader_is_reproducible_with_seed(self) -> None:
        first = self.reader.read()
        second = self.reader.read()

        np.testing.assert_allclose(first.train.features, second.train.features)
        np.testing.assert_allclose(first.train.labels, second.train.labels)
        np.testing.assert_allclose(first.test.features, second.test.features)
        np.testing.assert_allclose(first.test.labels, second.test.labels)

    def test_informative_pattern_builds_expected_coefficients(self) -> None:
        config = LinearSeparableDatasetConfig(size=1, features=13, informative=4)
        coefficients = config.build_coefficient_vector()

        expected_indices = np.array([2, 5, 8, 11])
        assert np.count_nonzero(coefficients) == 4
        np.testing.assert_array_equal(
            coefficients[expected_indices],
            np.ones_like(expected_indices, dtype=float),
        )

        zero_indices = np.setdiff1d(np.arange(config.features), expected_indices)
        np.testing.assert_array_equal(
            coefficients[zero_indices],
            np.zeros_like(zero_indices, dtype=float),
        )
