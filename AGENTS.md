# 開発ルール（LS-SVM 実証実験プロジェクト）

この文書は `fs4c` リポジトリで LS-SVM の実証実験を効率良く進めるための共通ルールをまとめたものです。README とコード構成を踏まえたプロジェクト固有の指針とします。

## 環境とツール
- Python 3.13 / `uv` を前提とし、依存追加後は `uv sync` を実行する。
- コマンド実行は基本的に `poe` タスクを利用（例: `poe main`）。新たなスクリプトは `tool.poe.tasks` に登録する。
- フォーマット・リンタ: `ruff`、型チェック: `pyright`。コミット前に `poe fmt` または `poe lint` を必ず実行。
- ログは `loguru` を用い、`utils.logging.setup_logger` を必ず呼び出す。

## レイヤードアーキテクチャ遵守
- 依存方向は README の Mermaid 図に従う。循環や逆方向の import を禁止。
- **Main/CLI (`main.py`)**: エントリポイントのみ。設定・CLI 引数の取得と Adapter 呼び出しに責務を限定。
- **Adapter/Scripts**: CLI やバッチスクリプトを配置。ユースケース層との橋渡し。外部 I/O/UI はここで完結させる。
- **App**: ビジネスロジックを担当。`ls_svm` 配下に UseCase と Service を実装し、Infra・Domain を利用して訓練/評価処理をまとめる。
- **Domain**: データセットごとの契約定義と、メタ情報（特徴量数、前処理要件など）を保持。実データや合成データはここでスキーマ化。
- **Infra**: 実データ取得・ファイル入出力・合成データ生成を担当。`domain` の契約を満たすよう実装する。
- **Utils**: ロギング・パス解決などの横断関心事。新規ユーティリティ追加時は依存方向が崩れないかを確認。

## モジュールの依存関係
【重要】以下のような__init__.pyを作成して依存関係を解決することは禁止
【重要】srcをroot dirに指定(Jetbrainsのeditorの機能)していることから、すべてのimportはsrc直下から
スタート可能になっている。つまりapp/???/、やdomain/???といった形のimportで参照できる。
```aiignore
Added src/app/service/ls_svm/__init__.py (+10 -0)
1     +"""LS-SVM サービスの公開モジュール。"""
2     +
3     +from .service import LSSVMHyperParameters, LSSVMModel, LSSVMService
4     +
5     +__all__ = [
6     +    "LSSVMHyperParameters",
7     +    "LSSVMModel",
8     +    "LSSVMService",
9     +]
10    +
```
【重要】import gurobi等の外部ライブラリのimportに関してif文分岐は必要ない。
理由 : importできない時点で 使用している箇所でエラーが出るから原因がわかるため。

## データセット運用
- 線形分離可能な合成データ、two-spiral、Iris/MNIST など 2 クラスデータを優先的に整備する。
- 合成データは `domain/dataset/<name>/contract.py` でスキーマや生成パラメータを定義し、`infra/dataset/read.py` から生成関数を提供する。
- 外部データを使用する場合は、ダウンロード手順・保存先・前処理を `README` または `adapter/scripts/README.md` に追記し、再現性を確保する。

## LS-SVM 実装指針
- 初期実装は 2 クラス分類に限定。ラベルは ±1 を基本とし、生成時に整形する。
- カーネル行列計算・線形方程式解法を `app/service/ls_svm/service.py` に集約。`gurobipy` を利用するが、必要に応じて数値線形代数でのフォールバック（`numpy.linalg.solve` など）も提供する。
- ハイパーパラメータ（γ、カーネル種別、σ など）は UseCase 層で受け取り Service に渡す。Adapter 層では CLI から指定できるようにする。
- モデル保存・読み込みが必要になった場合は `infra` に永続化ロジックを追加し、UseCase 層から呼び出す。

## 実験フロー
1. Adapter 層のスクリプトでデータロード → 前処理 → UseCase 呼び出し → 結果保存までを一気通貫で実装。
2. 評価指標（正解率、混同行列など）は UseCase 層で算出し、JSON/CSV などに出力する。
3. 実験設定（パラメータ、実行コマンド、結果出力パス）は `adapter/scripts/README.md` に記録する。
4. 可視化が必要な場合は Notebook ではなく Python スクリプトとして Adapter 層に配置。

## コーディング規約
- 型ヒントは必須。pydantic モデルなどデータ構造は `BaseModel` を使用。
- コメントは必要最小限（複雑なロジックや数式の説明など）に留める。
- ファイル名・モジュールは英小文字スネークケース。クラス名はパスカルケース。
- 定数や設定値は dotenv ではなく設定モジュール／pydantic-settings に集約予定。追加時は `pyproject.toml` へ依存を記載。

## ドキュメント更新
[docsフォルダ](./docs)には会話で確定した内容を記録していく。特にデータの生成規則の話、実行計画等で確定した内容を
開発者の合意とともに作成する ファイル名の命名規則は以下↓
`20251001_????.md`

以上のルールを遵守し、効率的かつ再現性の高い開発体制を維持する。
