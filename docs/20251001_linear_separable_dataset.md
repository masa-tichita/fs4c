# 線形分離可能データセット作成記録

## 背景
- LS-SVM 分類実装の初期検証に用いる人工データセットとして、既存論文のシミュレーション設定（多変量正規分布・係数パターン）を流用。
- 回帰向け設定だったため、判別値の符号で ±1 ラベルへ変換し、分類問題に適用できるよう調整。

## 実装概要
- `src/domain/dataset/linear_separable/contract.py`
  - `LinearSeparableDatasetConfig`: サンプル数(`size`)、特徴量数(`features`)、相関係数、ノイズ標準偏差、シードを保持。
  - デフォルトは全特徴量を均等重み（1.0）で利用。`informative` を指定した場合のみ (0,0,1) パターンを構築し、`features` に収まるようバリデーションを実施。
  - `LinearSeparableDataset`: 生成済みの特徴量・ラベル・係数を保持し、形状整合性を検証。
- `src/infra/dataset/read.py`
  - Toeplitz 型共分散 `Σ_ij = correlation^{|i-j|}` を生成。
  - `numpy.random.default_rng` で多変量正規サンプルを生成し、`noise_std` に従うノイズを加算。
  - 判別値 `features @ coefficients + noise` の符号で `±1` のラベルに変換し、`LinearSeparableDataset` を返却。

## 既定パラメータ
- `size = 100`
- `features = 25`
- `informative = None`（全特徴量を使用）
- `correlation = 0.35`
- `noise_std = 0.0`
- `seed = None`

## 利用手順
1. `uv sync` を実行し、追加した `numpy` 依存をインストール。
2. 以下のように呼び出してデータを生成：
   ```python
   from domain.dataset.linear_separable.contract import LinearSeparableDatasetConfig
   from infra.dataset.linear_separable.read import LinearSeparableDatasetReader

   config = LinearSeparableDatasetConfig(seed=42, noise_std=0.1)
   dataset = LinearSeparableDatasetReader(config).read()
   ```
3. 返却される `dataset.features` は `(size, features)`、`dataset.labels` は `±1` の 1 次元配列。

## 今後の予定
- 可視化スクリプトを Adapter 層に追加し、データ分布と境界の確認を行う。
- two-spiral や Iris など他の 2 クラスデータ契約も順次整備する。
