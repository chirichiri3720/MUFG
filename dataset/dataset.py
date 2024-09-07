import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .utils import feature_name_combiner
# make_calculate_two_features

logger = logging.getLogger(__name__)


# Copied from https://github.com/pfnet-research/deep-table.
# Modified by somaonishi and shoyameguro.
class TabularDataFrame(object):
    columns = []
    continuous_columns = []
    categorical_columns = []
    binary_columns = []
    target_column = "not.fully.paid"

    def __init__(
        self,
        seed,
        categorical_encoder="ordinal",
        continuous_encoder: str = None,
        **kwargs,
    ) -> None:
        """
        Args:
            root (str): Path to the root of datasets for saving/loading.
            download (bool): If True, you must implement `self.download` method
                in the child class. Defaults to False.
        """
        self.seed = seed
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder

        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))
        self.id = self.test.iloc[:,0]

        self.train = self.train[self.columns + [self.target_column]]
        self.test = self.test[self.columns]

        self.label_encoder = LabelEncoder().fit(self.train[self.target_column])
        # self.train[self.target_column] = self.label_encoder.transform(self.train[self.target_column])

    def _init_checker(self):
        variables = ["continuous_columns", "categorical_columns", "binary_columns", "target_column", "data"]
        for variable in variables:
            if not hasattr(self, variable):
                if variable == "data":
                    if not (hasattr(self, "train") and hasattr(self, "test")):
                        raise ValueError("TabularDataFrame does not define `data`, but neither does `train`, `test`.")
                else:
                    raise ValueError(f"TabularDataFrame does not define a attribute: `{variable}`")

    def show_data_details(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        logger.info(f"Dataset size       : {len(all_data)}")
        logger.info(f"All columns        : {all_data.shape[1] - 1}")
        logger.info(f"Num of cate columns: {len(self.categorical_columns)}")
        logger.info(f"Num of cont columns: {len(self.continuous_columns)}")

        y = all_data[self.target_column]
        class_ratios = y.value_counts(normalize=True)
        for label, class_ratio in zip(class_ratios.index, class_ratios.values):
            logger.info(f"class {label:<13}: {class_ratio:.3f}")

    def get_classify_dataframe(self) -> Dict[str, pd.DataFrame]:
        train = self.train
        test = self.test
        self.data_cate = pd.concat([train[self.categorical_columns], test[self.categorical_columns]])

        self.show_data_details(train, test)
        classify_dfs = {
            "train": train,
            "test": test,
        }
        return classify_dfs

    def fit_feature_encoder(self, df_train):
        # Categorical values are fitted on all data.
        if self.categorical_columns != []:
            if self.categorical_encoder == "ordinal":
                self._categorical_encoder = OrdinalEncoder(dtype=np.int32).fit(self.data_cate)
            elif self.categorical_encoder == "onehot":
                self._categorical_encoder = OneHotEncoder(
                    sparse_output=False,
                    feature_name_combiner=feature_name_combiner,
                    dtype=np.int32,
                    drop="first"
                ).fit(self.data_cate)
            else:
                raise ValueError(self.categorical_encoder)
        if self.continuous_columns != [] and self.continuous_encoder is not None:
            if self.continuous_encoder == "standard":
                self._continuous_encoder = StandardScaler()
            elif self.continuous_encoder == "minmax":
                self._continuous_encoder = MinMaxScaler()
            else:
                raise ValueError(self.continuous_encoder)
            self._continuous_encoder.fit(df_train[self.continuous_columns])
            #  self._continuous_encoder.fit_transform(df_train[self.continuous_columns])

    def apply_onehot_encoding(self, df: pd.DataFrame):
        encoded = self._categorical_encoder.transform(df[self.categorical_columns])
        encoded_columns = self._categorical_encoder.get_feature_names_out(self.categorical_columns)
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
        df = df.drop(self.categorical_columns, axis=1)
        return pd.concat([df, encoded_df], axis=1)

    def apply_feature_encoding(self, dfs):
        for key in dfs.keys():
            if self.categorical_columns != []:
                if isinstance(self._categorical_encoder, OrdinalEncoder):
                    dfs[key][self.categorical_columns] = self._categorical_encoder.transform(
                        dfs[key][self.categorical_columns]
                    )
                else:
                    dfs[key] = self.apply_onehot_encoding(dfs[key])
            if self.continuous_columns != []:
                if self.continuous_encoder is not None:
                    dfs[key][self.continuous_columns] = self._continuous_encoder.transform(
                        dfs[key][self.continuous_columns]
                    )
                else:
                    dfs[key][self.continuous_columns] = dfs[key][self.continuous_columns].astype(np.float64)
        if self.categorical_columns != []:
            if isinstance(self._categorical_encoder, OneHotEncoder):
                self.categorical_columns = self._categorical_encoder.get_feature_names_out(self.categorical_columns)
        return dfs

    def processed_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            dict[str, DataFrame]: The value has the keys "train", "val" and "test".
        """
        self._init_checker()
        dfs = self.get_classify_dataframe()
        # preprocessing
        self.fit_feature_encoder(dfs["train"])
        dfs = self.apply_feature_encoding(dfs)
        self.all_columns = list(self.categorical_columns) + list(self.continuous_columns) + list(self.binary_columns)
        return dfs

    def get_categories_dict(self):
        if not hasattr(self, "_categorical_encoder"):
            return None

        categories_dict: Dict[str, List[Any]] = {}
        for categorical_column, categories in zip(self.categorical_columns, self._categorical_encoder.categories_):
            categories_dict[categorical_column] = categories.tolist()

        return categories_dict
    

class V0(TabularDataFrame):
    continuous_columns = [
        "int.rate",
        # "installment",
        # "annual.inc",
        "dti",
        # "fico",
        # "days.with.cr.line",
        # "revol.bal",
        # "revol.util",
        "inq.last.6mths",
        # "delinq.2yrs",
        # "pub.rec",
        'fico_plus_revol.util',
        'fico_minus_revol.util',
        'fico_multiplied_by_revol.util',
        'fico_divided_by_revol.util',
        'fico_mean_revol.util',
        'fico_median_revol.util',
        'fico_q75_revol.util',
        'fico_q25_revol.util',
        'fico_zscore_f1_revol.util',
        'fico_plus_int.rate',
        'fico_minus_int.rate',
        'fico_multiplied_by_int.rate',
        'fico_divided_by_int.rate',
        'fico_mean_int.rate',
        'fico_median_int.rate',
        'fico_q75_int.rate',
        'fico_q25_int.rate',
        'fico_zscore_f1_int.rate',
    ]
    categorical_columns = [
        "purpose",
    ]
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train = pd.read_csv(to_absolute_path("datasets/train_fix.csv"))
        # self.train[self.target_column] = self.label_encoder.transform(self.train[self.target_column])
        self.test = pd.read_csv(to_absolute_path("datasets/test_fix.csv"))

class V1(TabularDataFrame):
    continuous_columns = [
        "int.rate",
        "installment",
        # "annual.inc",
        "dti",
        # "fico",
        "days.with.cr.line",
        "revol.bal",
        "revol.util",
      
        'installment_plus_annual.inc',
        'installment_minus_annual.inc',
        'installment_multiplied_by_annual.inc',
        'installment_divided_by_annual.inc',
        'installment_q75_annual.inc',
        'installment_zscore_f1_annual.inc',

        'fico_minus_revol.util',
        'fico_divided_by_revol.util',
        'fico_median_revol.util',
        'fico_q75_revol.util',
        'fico_q25_revol.util',
        'fico_zscore_f1_revol.util',

        'fico_minus_int.rate',
        'fico_multiplied_by_int.rate',
        'fico_divided_by_int.rate',
        'fico_mean_int.rate',
        'fico_zscore_f1_int.rate',
        
        'int.rate_plus_revol.util',
        'int.rate_minus_revol.util',
        'int.rate_multiplied_by_revol.util',
        'int.rate_divided_by_revol.util',
        'int.rate_q75_revol.util',
        'int.rate_q25_revol.util',
        'int.rate_zscore_f1_revol.util',

      
    ]
    categorical_columns = [
        'delinq.2yrs_0', 
        'delinq.2yrs_1',
        'delinq.2yrs_2',
        'delinq.2yrs_3over',
        'pub.rec_0', 
        'pub.rec_1',
        'inq_bin_div_0',
        'inq_bin_div_1',
        'inq_bin_div_2',
        'inq_bin_div_3',
        'inq_bin_div_4-',

        'installment_0-100',
        'installment_100-200',
        'installment_200-300',
        'installment_300-400',
        'installment_400-600',
        'installment_600-800',
        'installment_800over',

        # 'dti_0-5',
        # # 'dti_5-10',
        # # 'dti_10-15',
        # # 'dti_15-20',
        # # 'dti_20-25',
        # 'dti_0-5',
        # 'dti_5-20',
        # 'dti_20-25',
        # 'dti_25over',
        # # 'dti_25over',

        'int.rate_0',
        'int.rate_0_05',
        'int.rate_005_01',
        'int.rate_01_015',
        'int.rate_015_02',
        'int.rate_02_025',
        'int.rate_025over',

        'annual_0-20t',
        'annual_20t-40t',
        'annual_40t-60t',
        'annual_60t-80t',
        'annual_80t-100t',
        'annual_100t-120t',
        'annual_120t-140t',
        'annual_140t-',

        'fico_0-650',
        'fico_650-675',
        'fico_675-700',
        'fico_700-725',
        'fico_725-750',
        'fico_750-775',
        'fico_775-800',
        'fico_800-',

        'days_0-2500',
        'days_2500-5000',
        'days_5000-10000',
        # 'days_10000-15000',
        'days_10000-',

        'revol.bal0-5000',
        'revol.bal5000-10000',
        'revol.bal10000-15000',
        'revol.bal15000-20000',
        'revol.bal20000-',
    
        'revol.util0-20',
        'revol.util20-40',
        'revol.util40-60',
        'revol.util60-80',
        'revol.util80-100',
        'revol.util100-',
        



        'purpose',
    ]
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train = pd.read_csv(to_absolute_path("datasets/train_fix.csv"))
        # self.train[self.target_column] = self.label_encoder.transform(self.train[self.target_column])
        self.test = pd.read_csv(to_absolute_path("datasets/test_fix.csv"))