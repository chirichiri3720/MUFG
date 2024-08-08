import pandas as pd
import numpy as np
import statistics as st
import random

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])

train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]
test = test.drop('Transported', axis=1)

# csvファイルの作成
train.to_csv('datasets/train_fix3.csv', index=False)
test.to_csv('datasets/test_fix3.csv', index=False)
