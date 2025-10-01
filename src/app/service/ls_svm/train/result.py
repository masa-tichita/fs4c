from __future__ import annotations

from dataclasses import dataclass

from domain.model.ls_svm.contract import LSSVMModel


@dataclass(frozen=True)
class TrainLSSVMResult:
    model: LSSVMModel
