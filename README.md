# MUFG

## 環境設定
- python 3.10.6

poetryで管理してるので，以下でパッケージをインストールする．
```bash
poetry install
```

## 実行
```bash
python main.py
```

## 手順
- clone方法
```bash
git clone URL
```

- poetry作成
```bash
cd project_xyz
```
```bash
poetry init
```
```bash
poetry install
```
```bash
poetry add <package-name>
```
```bash
poetry add scikit-learn
poetry add hydra-core
poetry add optuna
poetry add xgboost
poetry add lightgbm
```
```bash
poetry update
```

- git 使い方

自分のブランチで作業する。
ブランチの作成方法
```bash
git branch "branch-name"
git checkout "branch-name"
```

コミットまで
```bash
git add "filename"
git commit -m "メッセージ"
git push origin "branch-name"
```

このタイミングでプルリクを出す

プルリクが通った後、
pullやり方
```bash
git checkout main
git pull origin main --rebase
git checkout "branch-name"
git pull origin main
```

- DATA

datasetsディレクトリを作成し、その中にサンプルデータを入れる。

preprocess.pyを使用し、データ処理

