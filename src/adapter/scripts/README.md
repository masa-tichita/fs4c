# Adapter Scripts

## LS-SVM 分類結果の可視化

`ls_svm_view.py` は線形分離可能な合成データを生成し、LS-SVM を学習して分類結果を比較する散布図を保存します。

```
poe run-visualize
```

または直接実行する場合:

```
uv run python -m adapter.scripts.ls_svm_view --size 200 --snr 1.0 --gamma 10.0
```

デフォルトでは 2 次元特徴量のデータセットを生成し、`artifacts/ls_svm_visualization.png` に真のラベルと推論ラベルを並べて保存します。
