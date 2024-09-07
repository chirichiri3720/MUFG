import numpy as np
import pandas as pd
from itertools import combinations


def feature_name_combiner(col, value) -> str:
    def replace(s):
        return s.replace("<", "lt_").replace(">", "gt_").replace("=", "eq_").replace("[", "lb_").replace("]", "ub_")

    col = replace(str(col))
    value = replace(str(value))
    return f'{col}="{value}"'


def feature_name_restorer(feature_name) -> str:
    return (
        feature_name.replace("lt_", "<").replace("gt_", ">").replace("eq_", "=").replace("lb_", "[").replace("ub_", "]")
    )


def label_encode(y: pd.Series):
    value_counts = y.value_counts(normalize=True)
    label_mapping = {value: index for index, (value, _) in enumerate(value_counts.items())}
    y_labels = y.map(label_mapping).astype(np.int32)
    return y_labels

def calculate_std(x, y):
    return pd.concat([x, y], axis=1).std(axis=1, ddof=1)

def calculate_mean(x, y):
    return (x + y) / 2

def calculate_median(x, y):
    return pd.concat([x, y], axis=1).median(axis=1)

def calculate_q75(x, y):
    return pd.concat([x, y], axis=1).quantile(0.75, axis=1)

def calculate_q25(x, y):
    return pd.concat([x, y], axis=1).quantile(0.25, axis=1)

def calculate_zscore(x, mean, std):
    return (x - mean) / (std + 1e-3)

def make_calculate_two_features(train, test, continuous_columns):
    df = pd.concat([train, test])

    feature_columns = df[[
        "int.rate",
        "dti",
        "fico",
        "days.with.cr.line",
        "inq.last.6mths",
        "delinq.2yrs",
        "pub.rec"
        ]]

    new_features = []
    for (feature1, feature2) in combinations(feature_columns, 2):
        f1, f2 = df[feature1], df[feature2]

        # 既存の特徴量操作
        new_features.append(f1 + f2)
        new_features.append(f1 - f2)
        new_features.append(f1 * f2)
        new_features.append(f1 / (f2 + 1e-8))

        # # 新しい特徴量操作
        new_features.append(calculate_mean(f1, f2))
        new_features.append(calculate_median(f1, f2))
        new_features.append(calculate_q75(f1, f2))
        new_features.append(calculate_q25(f1, f2))
        zscore_f1 = calculate_zscore(
            f1, calculate_mean(f1, f2), calculate_std(f1, f2))
        new_features.append(zscore_f1)

    new_features_df = pd.concat(new_features, axis=1)

    # カラム名の更新
    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                # for operation in ['multiplied_by', 'divided_by', 'q75', 'q25', 'zscore_f1']]
                                for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'mean', 'median', 'q75', 'q25', 'zscore_f1']]

    continuous_columns.extend(new_features_df.columns)
    
    result_df = pd.concat([df, new_features_df], axis=1)

    train = result_df.iloc[:len(train)]
    test = result_df.iloc[len(train):]

    return train, test, continuous_columns