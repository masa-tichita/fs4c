from __future__ import annotations

from typing import cast

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from app.service.ls_svm.common.kernel import resolve_kernel_function
from app.service.ls_svm.train.args import TrainLSSVMArgs
from app.service.ls_svm.train.result import TrainLSSVMResult
from domain.model.ls_svm.contract import LSSVMHyperParameters, LSSVMModel


class TrainLSSVMService:
    """LS-SVM の学習ロジックを提供するサービス。"""

    def execute(self, args: TrainLSSVMArgs) -> TrainLSSVMResult:
        features = np.asarray(args.features, dtype=np.float64)
        labels = np.asarray(args.labels, dtype=np.float64)

        kernel_matrix = self._build_kernel_matrix(
            args.hyperparameters, lhs=features, rhs=features
        )
        system_matrix, rhs = self._build_linear_system(
            kernel_matrix, labels, args.hyperparameters.gamma
        )
        solution = self._solve_linear_system(system_matrix, rhs)

        bias = float(solution[0])
        alphas = solution[1:]

        model = LSSVMModel(
            bias=bias,
            alphas=alphas,
            support_vectors=features,
            support_labels=labels,
            hyperparameters=args.hyperparameters,
        )
        return TrainLSSVMResult(model=model)

    def _build_linear_system(
        self,
        kernel_matrix: NDArray[np.float64],
        labels: NDArray[np.float64],
        gamma: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_samples = labels.shape[0]
        omega = (labels[:, None] * labels[None, :]) * kernel_matrix

        system = np.zeros((n_samples + 1, n_samples + 1), dtype=np.float64)
        system[0, 1:] = labels
        system[1:, 0] = labels
        system[1:, 1:] = omega + (1.0 / gamma) * np.eye(n_samples)

        rhs = np.zeros(n_samples + 1, dtype=np.float64)
        rhs[1:] = 1.0
        return system, rhs

    def _solve_linear_system(
        self, matrix: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        try:
            solved = np.linalg.solve(matrix, rhs)
            return cast(NDArray[np.float64], solved)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            msg = "LS-SVM の線形方程式を解けませんでした。"
            raise RuntimeError(msg) from exc

    def _solve_with_gurobi(
        self, matrix: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        model = gp.Model("ls_svm_linear_system")
        model.Params.OutputFlag = 0

        vars_dim = rhs.shape[0]
        variables = model.addMVar(shape=vars_dim, lb=-GRB.INFINITY, name="solution")
        var_list = [variables[j] for j in range(vars_dim)]

        for idx in range(matrix.shape[0]):
            coeffs = matrix[idx, :]
            expr = gp.LinExpr(coeffs.tolist(), var_list)  # type: ignore[arg-type]
            model.addConstr(expr == float(rhs[idx]))

        model.setObjective(0.0, GRB.MINIMIZE)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            msg = "Gurobi で線形方程式が最適化されませんでした。"
            raise RuntimeError(msg)

        solution = np.array(variables.X, dtype=np.float64)
        return cast(NDArray[np.float64], solution)

    def _build_kernel_matrix(
        self,
        hyperparameters: LSSVMHyperParameters,
        lhs: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        kernel_fn = resolve_kernel_function(hyperparameters)
        return kernel_fn(lhs, rhs)
