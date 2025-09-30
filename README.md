# fs4c
- fs4c(feature selection for classification)
- 分類のための特徴量選択手法を比較するためのリポジトリ


## Environment
- Python3.13
- uv

## Installation

```
uv sync
```

## development
- main.pyの実行
```
poe main
```

## Architecture
- レイヤードアーキテクチャを採用
- 以下の依存関係で開発を順守しながら開発を進める

```mermaid
graph LR
%% ===== Layers =====
subgraph MAIN["Main / CLI"]
    A[main.py]
end

subgraph ADAPTER["Adapter / Scripts"]
    B[adapter/scripts/*]
end

subgraph APP["App - UseCases & Services"]
    C1[app/ls_svm/train/usecase.py]
    C2[app/ls_svm/evaluate/usecase.py]
    C3[app/service/ls_svm/service.py]
end

subgraph DOMAIN["Domain - Dataset Contracts"]
    D1[domain/dataset/iris/contract.py]
    D2[domain/dataset/mnist/contract.py]
end

subgraph INFRA["Infra - IO"]
    E1[infra/dataset/read.py]
end

subgraph UTILS["Utils"]
    U1[utils/logging.py]
    U2[utils/system.py]
end

%% ===== Allowed Dependencies =====
A --> B
A --> U1
A --> U2

B --> C1
B --> C2
B --> U1
B --> U2

C1 --> C3
C1 --> D1
C1 --> D2
C1 --> E1
C1 --> U1
C1 --> U2

C2 --> C3
C2 --> D1
C2 --> D2
C2 --> E1
C2 --> U1
C2 --> U2

C3 --> D1
C3 --> D2
C3 --> U1
C3 --> U2

E1 --> D1
E1 --> D2
```