# LS-SVM サービス実装メモ

## 実装概要
- `app/service/ls_svm/train/service.py` に LS-SVM の学習ロジックを集約し、入力 (`args.py`) / 出力 (`result.py`) を明示。
- 推論処理は `app/service/ls_svm/predict/service.py` に分離し、同様に `args.py` / `result.py` で契約化。
- ハイパーパラメータおよび学習結果の型は `domain/model/ls_svm/contract.py` に移管（`LSSVMHyperParameters` / `LSSVMModel`）。
- 解法は Gurobi による線形方程式求解を優先し、例外時には `numpy.linalg.solve` へフォールバック。
- カーネル計算ロジックは `app/service/ls_svm/common/kernel.py` に切り出し、学習・推論サービスで共通利用。
- 学習結果は `LSSVMModel` として返却し、`app/ls_svm/usecase.py` から train/predict 両ユースケースを経由して利用。

## カーネル対応
- `linear`: 既定。内積による Gram 行列を算出。
- `rbf`: `rbf_sigma` を必須とし、ガウシアンカーネルで Gram 行列を構築。

## バリデーションと制約
- ラベルは ±1 のみ許容。
- `rbf` 選択時は `rbf_sigma` の指定が必須。

## テスト
- `tests/app/ls_svm/test_services.py` で学習・推論サービスを検証。
- `tests/app/ls_svm/test_usecases.py` でユースケースを通じた一貫性を確認。
- `src/adapter/scripts/ls_svm_visualize.py` を追加し、分類結果の可視化を自動生成できるようにした。
