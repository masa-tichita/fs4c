# 概要
- docsは実行計画やデータの作成手順等を記録していくもの

## ルール
- 基本はcodexとの対話を行う → codexが自動生成する

## 合成データセット作成記録
- 目的: LS-SVM 分類実験の基礎検証用に線形分離可能な人工データを用意
- 実装ファイル:
  - `src/domain/dataset/linear_separable/contract.py` : 生成パラメータとデータ構造の契約を定義
  - `src/infra/dataset/read.py` : 多変量正規分布からサンプルを生成し、`sign(a*^T x + ε)` で ±1 ラベル化
- 生成仕様:
  - 既定値は `size=100`, `features=25`, `informative=None`, 相関 `Σ_ij = 0.35^{|i-j|}`
  - 真の係数ベクトルは既定で全特徴量を均等重み（1.0）とし、必要に応じて `informative` を指定して (0,0,1) パターンへ切り替える。`seed` と `noise_std` で再現性とノイズ量を制御
  - 補足: `pyproject.toml` に `numpy` 依存を追加済み。`uv sync` 実行後、`LinearSeparableDatasetReader` で再現可能
