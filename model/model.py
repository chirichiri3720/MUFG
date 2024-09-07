import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import pandas as pd

from sklearn.utils.validation import check_X_y

from sklearn.linear_model import LogisticRegression

from .base_model import BaseClassifier, BaseRegressor
from .utils import f1_micro, f1_micro_lgb, binary_logloss


class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",  # 2値分類用のobjectiveに変更
            eval_metric="logloss",  # 2値分類なのでloglossを使用
            early_stopping_rounds=100,
            **self.model_config,
            random_state=seed
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(X, y, eval_set=eval_set, verbose=self.verbose > 0)
        # print("XGBoostモデルの使用されたパラメータ:", self.model.get_params())

    def feature_importance(self):
        return self.model.feature_importances_

class LightGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMClassifier(
            objective="binary",  # 2値分類用のobjectiveに変更
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=binary_logloss,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0)],
        )
        # print("LightGBMモデルの使用されたパラメータ:", self.model.get_params())
        
    def feature_importance(self):
        return self.model.feature_importances_


class XGBoostRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",  # 目的関数を回帰用に変更
            eval_metric="rmse",             # 評価指標を回帰用に変更
            early_stopping_rounds=50,
            **self.model_config,
            random_state=seed
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(X, y, eval_set=eval_set, verbose=self.verbose > 0)
        # print("XGBoostモデルの使用されたパラメータ:", self.model.get_params())

    def feature_importance(self):
        return self.model.feature_importances_

class LightGBMRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMRegressor(
            objective="regression",         # 目的関数を回帰用に変更
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric="rmse",                    # 評価指標を回帰用に変更
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0)],
        )
        # print("LightGBMモデルの使用されたパラメータ:", self.model.get_params())
        
    def feature_importance(self):
        return self.model.feature_importances_

class CatBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = cat.CatBoostClassifier(
            loss_function='Logloss',
            # loss_function='CrossEntropy',  # 損失関数を設定
            # use_best_model=True,
            # early_stopping_rounds=500,
            early_stopping_rounds=100, #optunaはこっち
            **model_config,
            random_seed=seed,
            eval_metric="AUC",
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_pool = (eval_set[0], eval_set[1])  # eval_setを適切な形式に変更

        self.model.fit(X, y, eval_set=eval_pool, verbose=self.verbose > 0)

    def feature_importance(self):
        return self.model.get_feature_importance()
    
class CatBoostRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = cat.CatBoostClassifier(
            loss_function='Logloss',
            # loss_function='CrossEntropy',  # 損失関数を設定
            # use_best_model=True,
            # early_stopping_rounds=500,
            early_stopping_rounds=100, #optunaはこっち
            **model_config,
            random_seed=seed,
            eval_metric="AUC",
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        
        train_pool = cat.Pool(X, y)
        eval_pools = [cat.Pool(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, y_val) for X_val, y_val in eval_set]

        self.model.fit(
            train_pool,
            eval_set=eval_pools,
            use_best_model=True,
            early_stopping_rounds=75,
        )
        best_iteration = self.model.get_best_iteration()
        
        best_score = self.model.get_best_score()

        print(f"Best iteration: {best_iteration}")
        print(f"Best score: {best_score}")

    def feature_importance(self):
        return self.model.get_feature_importance()

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        
        #model_configからの設定を使用してロジスティック回帰を初期化
        self.model = LogisticRegression(**model_config)


    def fit(self, X, y, eval_set=None):
        self._column_names = X.columns
        # Xとyが適切な形状を持つことを確認
        X, y = check_X_y(X, y)
        
        # 訓練データでモデルを実装
        self.model.fit(X, y)
        
        if self.verbose > 0:
            print("Model fitted with training data.")

    def feature_importance(self):
        return self.model.coef_