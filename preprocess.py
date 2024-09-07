import pandas as pd
import numpy as np
import statistics as st
import random
from itertools import combinations

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])

def missing_value_checker(df, name):
    chk_null = df.isnull().sum()
    chk_null_pct = chk_null / (df.index.max() + 1)
    chk_null_tbl = pd.concat([chk_null[chk_null > 0], chk_null_pct[chk_null_pct > 0]], axis=1)
    chk_null_tbl = chk_null_tbl.rename(columns={0: "欠損数",1: "欠損割合"})
    print(name)
    print(chk_null_tbl, end="\n\n")

def random_completion_purpose(df, name):#purposeをランダムで補完
    categories = [
        'debt_consolidation', 
        'credit_card', 
        'all_other', 
        'small_business', 
        'home_improvement', 
        'educational', 
        'major_purchase'
    ]

    np.random.seed(93)

    # 欠損値の補完
    df[name] = df[name].apply(
        # lambda x: x if pd.notna(x) else np.random.choice(categories)
        lambda x: x if pd.notna(x) else 'debt_consolidation'
        # lambda x: x if pd.notna(x) else 'all_other'
    )
    return df

def average_completion_other(df):#その他の欠損値をaverageで補完
    target_columns = [
        'installment',
        'revol.bal',
        'revol.util',
    ]
    for column in target_columns:
        if column in df.columns:
            mean_value = df[column].mean()
            df[column] = df[column].fillna(mean_value)
    
    return df

def drop_columns(df):
    target_columns = [
        #  "int.rate",
        "installment",
        # "annual.inc",
        # "dti",
        # "fico",
        # "days.with.cr.line",
        # "revol.bal",
        # "revol.util",
    ]

    return df.drop(target_columns,axis=1)

def categorical_encoding(df):
    target_columns = [
        'inq.last.6mths',
        'delinq.2yrs',
        'pub.rec'
    ]

    df['inq_bin_div_0'] = ((df['inq.last.6mths'] == 0)).astype(int)
    df['inq_bin_div_1'] = ((df['inq.last.6mths'] == 1)).astype(int)
    df['inq_bin_div_2'] = ((df['inq.last.6mths'] == 2)).astype(int)
    df['inq_bin_div_3'] = ((df['inq.last.6mths'] == 3)).astype(int)
    df['inq_bin_div_4-'] = ((df['inq.last.6mths'] >= 4)).astype(int)
    # df['inq_bin_div_5-8'] = ((df['inq.last.6mths'] >4) & (df['inq.last.6mths'] <= 8)).astype(int)
    # df['inq_bin_div_8-10'] = ((df['inq.last.6mths'] >7) & (df['inq.last.6mths'] <= 10)).astype(int)
    # df['inq_bin_div_9-'] = ((df['inq.last.6mths'] >= 9)).astype(int)

    df['delinq.2yrs_0'] = ((df['delinq.2yrs'] <= 0)).astype(int)
    df['delinq.2yrs_1'] = ((df['delinq.2yrs']  > 0) & (df['delinq.2yrs'] <= 1)).astype(int)
    df['delinq.2yrs_2'] = ((df['delinq.2yrs']  > 1) & (df['delinq.2yrs'] <= 2)).astype(int)
    df['delinq.2yrs_3over'] = ((df['delinq.2yrs']  > 2)).astype(int)

    df['pub.rec_0'] = ((df['pub.rec'] == 0)).astype(int)
    df['pub.rec_1'] = ((df['pub.rec'] == 1)).astype(int)

    df = df.drop(target_columns,axis=1)

    return df

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

def conbination_columns(df):
    new_features = []
    feature_columns = df[['fico','revol.util']]
    for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = df[feature1], df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 + f2)
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))
    
            # # 新しい特徴量操作
            # new_features.append(calculate_mean(f1, f2))
            # new_features.append(calculate_median(f1, f2))
            # new_features.append(calculate_q75(f1, f2))
            # new_features.append(calculate_q25(f1, f2))
            # zscore_f1 = calculate_zscore(
            #     f1, calculate_mean(f1, f2), calculate_std(f1, f2))
            # new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                # for operation in ['multiplied_by', 'divided_by', 'q75', 'q25', 'zscore_f1']]
                                # for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'mean', 'median', 'q75', 'q25', 'zscore_f1']]
                                for operation in ['plus', 'minus', 'multiplied_by', 'divided_by']]
    
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def installment_and_annual_inc(df):
    new_features = []
    feature_columns = df[['installment','annual.inc']]
    for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = df[feature1], df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 + f2)
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))
    
            # # 新しい特徴量操作
            new_features.append(calculate_q75(f1, f2))
            zscore_f1 = calculate_zscore(
                f1, calculate_mean(f1, f2), calculate_std(f1, f2))
            new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'q75', 'zscore_f1']]
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def fico_and_int_rate(df):
    new_features = []
    feature_columns = df[['fico','int.rate']]
    for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = df[feature1], df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))
    
            # # 新しい特徴量操作
            new_features.append(calculate_mean(f1, f2))
            zscore_f1 = calculate_zscore(
                f1, calculate_mean(f1, f2), calculate_std(f1, f2))
            new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                for operation in ['minus', 'multiplied_by', 'divided_by','mean', 'zscore_f1']]
  
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def fico_and_revol_util(df):
    new_features = []
    feature_columns = df[['fico','revol.util']]
    for (feature1, feature2) in combinations(feature_columns, 2):
        f1, f2 = df[feature1], df[feature2]

        # 既存の特徴量操作
        new_features.append(f1 - f2)
        new_features.append(f1 / (f2 + 1e-8))
        # # 新しい特徴量操作
        new_features.append(calculate_median(f1, f2))
        new_features.append(calculate_q75(f1, f2))
        new_features.append(calculate_q25(f1, f2))
        zscore_f1 = calculate_zscore(
            f1, calculate_mean(f1, f2), calculate_std(f1, f2))
        new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                for operation in ['minus', 'divided_by','median', 'q75', 'q25','zscore_f1']]
    
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def int_rate_and_revol_util(df):
    new_features = []
    feature_columns = df[['int.rate','revol.util']]
    for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = df[feature1], df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 + f2)
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))
    
            # # 新しい特徴量操作
            new_features.append(calculate_q75(f1, f2))
            new_features.append(calculate_q25(f1, f2))
            zscore_f1 = calculate_zscore(
                f1, calculate_mean(f1, f2), calculate_std(f1, f2))
            new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'q75', 'q25', 'zscore_f1']]
    
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def installment_bin(df):
    target_columns = [
        'installment'
    ]

    df['installment_0-100'] = ((df['installment'] > 0) & (df['installment'] <= 100)).astype(int)
    df['installment_100-200'] = ((df['installment'] > 100 )& (df['installment'] <= 200)).astype(int)
    df['installment_200-300'] = ((df['installment'] > 200 )& (df['installment'] <= 300)).astype(int)
    df['installment_300-400'] = ((df['installment'] > 300 )& (df['installment'] <= 400)).astype(int)   
    df['installment_400-600'] = ((df['installment'] > 400 )& (df['installment'] <= 600)).astype(int)
    df['installment_600-800'] = ((df['installment'] > 600 )& (df['installment'] <= 800)).astype(int)
    df['installment_800over'] = ((df['installment'] > 800)).astype(int)

    # df = df.drop(target_columns,axis=1)

    return df

def dti_bin(df):
    target_columns = [
         'dti'
    ]
    # df['dti_0-5'] = ((df['dti'] > 0) & (df['dti'] <= 5)).astype(int)
    # df['dti_5-10'] = ((df['dti'] > 5) & (df['dti'] <= 10)).astype(int)
    # df['dti_10-15'] = ((df['dti'] > 10) & (df['dti'] <= 15)).astype(int)
    # df['dti_15-20'] = ((df['dti'] > 15) & (df['dti'] <= 20)).astype(int)
    # df['dti_20-25'] = ((df['dti'] > 20) & (df['dti'] <= 25)).astype(int)
    df['dti_0-5'] = ((df['dti'] > 0) & (df['dti'] <= 5)).astype(int)
    df['dti_5-20'] = ((df['dti'] > 5) & (df['dti'] <= 20)).astype(int)
    df['dti_20-25'] = ((df['dti'] > 20) & (df['dti'] <= 25)).astype(int)
    df['dti_25over'] = (df['dti'] > 25).astype(int)

    # df = df.drop(target_columns, axis=1)

    return df

def int_rate_bin(df):
    target_columns = [
         'int.rate'
    ]
    df['int.rate_0'] = ((df['int.rate'] == 0)).astype(int)
    df['int.rate_0_05'] = ((df['int.rate'] > 0) & (df['int.rate'] <= 0.05)).astype(int)
    df['int.rate_005_01'] = ((df['int.rate'] > 0.05) & (df['int.rate'] <= 0.1)).astype(int)
    df['int.rate_01_015'] = ((df['int.rate'] > 0.1) & (df['int.rate'] <= 0.15)).astype(int)
    df['int.rate_015_02'] = ((df['int.rate'] > 0.15) & (df['int.rate'] <= 0.2)).astype(int)
    df['int.rate_02_025'] = ((df['int.rate'] > 0.2) & (df['int.rate'] <= 0.25)).astype(int)
    df['int.rate_025over'] = (df['int.rate'] > 0.25).astype(int)

    # df = df.drop(target_columns, axis=1)

    return df

def annual_inc_bin(df):
    target_columns = [
         'annual.inc'
    ]
    df['annual_0-20t'] = ((df['annual.inc'] >= 0) & (df['annual.inc'] < 20000)).astype(int)
    df['annual_20t-40t'] = ((df['annual.inc'] >= 20000) & (df['annual.inc'] < 40000)).astype(int)
    df['annual_40t-60t'] = ((df['annual.inc'] >= 40000) & (df['annual.inc'] < 60000)).astype(int)
    df['annual_60t-80t'] = ((df['annual.inc'] >= 60000) & (df['annual.inc'] < 80000)).astype(int)
    df['annual_80t-100t'] = ((df['annual.inc'] >= 80000) & (df['annual.inc'] < 100000)).astype(int)
    df['annual_100t-120t'] = ((df['annual.inc'] >= 100000) & (df['annual.inc'] < 120000)).astype(int)
    df['annual_120t-140t'] = ((df['annual.inc'] >= 120000) & (df['annual.inc'] < 140000)).astype(int)
    df['annual_140t-'] = ((df['annual.inc'] >= 140000)).astype(int)

    df = df.drop(target_columns,axis = 1)

    return df

def fico_bin(df):
    target_columns = [
          'fico'
    ]
    # df['fico_0-650'] = ((df['fico'] >= 0) & (df['fico'] < 650)).astype(int)
    # df['fico_650-675'] = ((df['fico'] >= 650) & (df['fico'] < 675)).astype(int)
    # df['fico_675-700'] = ((df['fico'] >= 675) & (df['fico'] < 700)).astype(int)
    # df['fico_700-725'] = ((df['fico'] >= 700) & (df['fico'] < 725)).astype(int)
    # df['fico_725-750'] = ((df['fico'] >= 725) & (df['fico'] < 750)).astype(int)
    # df['fico_750-775'] = ((df['fico'] >= 750) & (df['fico'] < 775)).astype(int)
    # df['fico_775-'] = ((df['fico'] >= 775)).astype(int)

    
    df['fico_0-650'] = ((df['fico'] >= 0) & (df['fico'] < 650)).astype(int)
    df['fico_650-675'] = ((df['fico'] >= 650) & (df['fico'] < 675)).astype(int)
    df['fico_675-700'] = ((df['fico'] >= 675) & (df['fico'] < 700)).astype(int)
    df['fico_700-725'] = ((df['fico'] >= 700) & (df['fico'] < 725)).astype(int)
    df['fico_725-750'] = ((df['fico'] >= 725) & (df['fico'] < 750)).astype(int)
    df['fico_750-775'] = ((df['fico'] >= 750) & (df['fico'] < 775)).astype(int)
    df['fico_775-800'] = ((df['fico'] >= 775) & (df['fico'] < 800)).astype(int)
    df['fico_800-'] = ((df['fico'] >= 800)).astype(int)


    df = df.drop(target_columns,axis=1)

    return df

def days_with_bin(df):
    target_columns = [
          'days.with.cr.line'
     ]
    
    df['days_0-2500'] = ((df['days.with.cr.line'] >= 0) & (df['days.with.cr.line'] < 2500)).astype(int)
    df['days_2500-5000'] = ((df['days.with.cr.line'] >= 2500) & (df['days.with.cr.line'] < 5000)).astype(int)
    df['days_5000-10000'] = ((df['days.with.cr.line'] >= 5000) & (df['days.with.cr.line'] < 10000)).astype(int)
    # df['days_10000-15000'] = ((df['days.with.cr.line'] >= 10000) & (df['days.with.cr.line'] < 15000)).astype(int)
    df['days_10000-'] = ((df['days.with.cr.line'] >= 10000)).astype(int)
    # df = df.drop(target_columns,axis=1)

    return df

def revol_bal_bin(df):
    target_columns = [
          'revol.bal'
     ]
    
    df['revol.bal0-5000'] = ((df['revol.bal'] >= 0) & (df['revol.bal'] < 5000)).astype(int)
    df['revol.bal5000-10000'] = ((df['revol.bal'] >= 5000) & (df['revol.bal'] < 10000)).astype(int)
    df['revol.bal10000-15000'] = ((df['revol.bal'] >= 10000) & (df['revol.bal'] < 15000)).astype(int)
    df['revol.bal15000-20000'] = ((df['revol.bal'] >= 15000) & (df['revol.bal'] < 20000)).astype(int)
    df['revol.bal20000-'] = ((df['revol.bal'] >= 20000)).astype(int)
    # df = df.drop(target_columns,axis=1)

    return df

def revol_util_bin(df):
    target_columns = [
          'revol.util'
     ]
    
    df['revol.util0-20'] = ((df['revol.util'] >= 0) & (df['revol.util'] < 20)).astype(int)
    df['revol.util20-40'] = ((df['revol.util'] >= 20) & (df['revol.util'] < 40)).astype(int)
    df['revol.util40-60'] = ((df['revol.util'] >= 40) & (df['revol.util'] < 60)).astype(int)
    df['revol.util60-80'] = ((df['revol.util'] >= 60) & (df['revol.util'] < 80)).astype(int)
    df['revol.util80-100'] = ((df['revol.util'] >= 80) & (df['revol.util'] < 100)).astype(int)
    df['revol.util100-'] = ((df['revol.bal'] >= 100)).astype(int)

    # df = df.drop(target_columns,axis=1)

    return df

df = train_test

#欠損値の有無の確認
missing_value_checker(df, "before train_test")

df = random_completion_purpose(df, 'purpose')

df = average_completion_other(df)

df = installment_and_annual_inc(df)
df = fico_and_revol_util(df)
df = fico_and_int_rate(df)
df = int_rate_and_revol_util(df)


df = categorical_encoding(df)

df = installment_bin(df) 

# df = dti_bin(df)  #

df = days_with_bin(df) #

df = int_rate_bin(df)

df = annual_inc_bin(df)

df = fico_bin(df)

df = revol_bal_bin(df)

df = revol_util_bin(df)

# df = drop_columns(df)

df.to_csv('datasets/concat_fix.csv', index=False)

#欠損値があるか再確認
missing_value_checker(df,"after train_test")

print(df.info())
train_test = df

train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]

test = test.drop("not.fully.paid",axis=1)

# csvファイルの作成
train.to_csv('datasets/train_fix.csv', index=False)
test.to_csv('datasets/test_fix.csv', index=False)
