# pytest 設定調整記録

## 背景
- `poe test` 実行時に `asyncio_default_fixture_loop_scope` 等の pytest 設定に対する `PytestConfigWarning` が発生していた。
- `pyproject.toml` に既存で設定している asyncio 関連・タイムアウト設定を活かすため、該当プラグインが必要と判断。

## 対応内容
- 開発依存 (`[dependency-groups].dev`) に `pytest-asyncio>=0.24.0` と `pytest-timeout>=2.3.1` を追加。
- `UV_CACHE_DIR=.uv-cache uv run poe test` を実行し、プラグインが読み込まれることを確認。

## 確認結果
- pytest 実行時に `plugins: asyncio-1.2.0, timeout-2.4.0, cov-7.0.0` が読み込まれ、警告が消失。
- 既存のテスト（3 件）が全て成功し、追加の副作用なし。

