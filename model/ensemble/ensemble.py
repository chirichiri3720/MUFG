import numpy as np

from ..base_model import BaseClassifier
from ..model import LightGBMClassifier, XGBoostClassifier, CatBoostClassifier


class XGBLGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostClassifier(input_dim, output_dim, model_config=self.model_config.xgboost, verbose=0)
        self.lgbm_model = LightGBMClassifier(input_dim, output_dim, model_config=self.model_config.lightgbm, verbose=0)

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return (self.xgb_model.predict_proba(X) + self.lgbm_model.predict_proba(X)) / 2

    def feature_importance(self):
        return self.xgb_model.feature_importance(), self.lgbm_model.feature_importance()

class XGB10Classifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.models = []
        for i in range(10):
            xgb = XGBoostClassifier(input_dim, output_dim, model_config=self.model_config, verbose=0, seed=i)
            self.models.append(xgb)

    def fit (self, X, y, eval_set):
        for model in self.models:
            model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.mean([model.predict_proba(X) for model in self.models], axis=0)
        
    def feature_importance(self):
        return None

class XGB7LGBM7Classifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.models = []
        lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for i in range(7):
            self.model_config.xgboost.eta = lr[i]
            xgb = XGBoostClassifier(input_dim, output_dim, model_config=self.model_config.xgboost, verbose=0)
            self.models.append(xgb)
            self.model_config.lightgbm.learning_rate = lr[i]
            lgbm = LightGBMClassifier(input_dim, output_dim, model_config=self.model_config.lightgbm, verbose=0)
            self.models.append(lgbm)

    def fit(self, X, y, eval_set):
        for model in self.models:
            model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.mean([model.predict_proba(X) for model in self.models], axis=0)

    def feature_importance(self):
        return None

class XGBLRClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.models = []
        lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for i in range(7):
            self.model_config.eta = lr[i]
            xgb = XGBoostClassifier(input_dim, output_dim, model_config=self.model_config, verbose=0)
            self.models.append(xgb)

    def fit(self, X, y, eval_set):
        for model in self.models:
            model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.mean([model.predict_proba(X) for model in self.models], axis=0)

    def feature_importance(self):
        return None

class XGBLGBMCATClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostClassifier(input_dim, output_dim, model_config=self.model_config.xgboost, verbose=0)
        self.lgbm_model = LightGBMClassifier(input_dim, output_dim, model_config=self.model_config.lightgbm, verbose=0)
        self.cat_model = CatBoostClassifier(input_dim, output_dim, model_config=self.model_config.catboost, verbose=0)

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)
        self.cat_model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return (self.xgb_model.predict_proba(X) + self.lgbm_model.predict_proba(X) + self.cat_model.predict_proba(X)) / 3

    def feature_importance(self):
        return self.xgb_model.feature_importance(), self.lgbm_model.feature_importance(), self.cat_model.feature_importance()

class CATSEEDClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.models = []
        for i in range(10):
            cat = CatBoostClassifier(input_dim, output_dim, model_config=self.model_config, verbose=0, seed=i)
            self.models.append(cat)

    def fit (self, X, y, eval_set):
        for model in self.models:
            model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.mean([model.predict_proba(X) for model in self.models], axis=0)
        
    def feature_importance(self):
        return None

class CATLRClassifier(BaseClassifier):  #異なる学修率を持つcatboostのアンサンブル
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.models = []
        # lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        lr = [0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
        for i in range(14):
            self.model_config.learning_rate = lr[i]
            cat = CatBoostClassifier(input_dim, output_dim, model_config=self.model_config, verbose=0)
            self.models.append(cat)

    def fit(self, X, y, eval_set):
        for model in self.models:
            model.fit(X, y, eval_set)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.mean([model.predict_proba(X) for model in self.models], axis=0)

    def feature_importance(self):
        return None