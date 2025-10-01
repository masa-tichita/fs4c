# Adapter Scripts

## LS-SVM 分類結果の可視化

`ls_svm_view.py` は線形分離可能な合成データを生成し、LS-SVM を学習して分類結果を比較する散布図を保存します。学習と推論時間はユースケース側の `@log_fn` で計測されます。

```
poe run-view
```

または直接実行する場合:

```
uv run python -m adapter.scripts.ls_svm_view
```

デフォルトでは SNR=1.0・特徴量数 2・学習/評価比率 0.7 のデータセットを生成し、`artifacts/ls_svm_visualization.png` に訓練データの真値とテストデータの予測結果を並べて保存します。
