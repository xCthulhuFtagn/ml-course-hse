from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve, auc
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
            threshold: float = 0.6
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        if not 0 < learning_rate <= 1: raise ValueError("Learning rate must be in (0, 1]")
        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot
        self.threshold = threshold

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
    # bootstrap
        idx = np.random.choice(
            np.random.choice(np.arange(x.shape[0]),size=int(x.shape[0]*self.subsample), replace=False),
            size = x.shape[0],
            replace=True
        )
        new_model = self.base_model_class().fit(x[idx, :], -self.loss_derivative(y, predictions)[idx])
        self.gammas.append(self.find_optimal_gamma(y, predictions, new_model.predict(x)))
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid = None, y_valid = None):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        self.classes_ = np.unique(y_train)
        if x_valid is None or y_valid is None:
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.5, random_state=1337)

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        if self.early_stopping_rounds is not None: val_loss = np.ndarray((self.early_stopping_rounds,))
        for est_cnt in range(1, self.n_estimators + 1):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            
            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)
            
            if self.early_stopping_rounds is not None:
                val_loss[(est_cnt-1) % self.early_stopping_rounds] = self.loss_fn(y_valid, valid_predictions)
                if est_cnt % self.early_stopping_rounds == 0:
                    if (self.validation_loss == val_loss).all(): break
                    else: self.validation_loss = val_loss
        if self.plot:
            fpr, tpr, _ = roc_curve(y_train, train_predictions)
            roc_auc = auc(fpr, tpr)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        return self
            
    def predict_proba(self, x):
        pred = 0
        for gamma, model in zip(self.gammas, self.models):
            pred += gamma * model.predict(x)
        pred = self.sigmoid(pred)
        # for some reason pred is scalar and sometimes it's not
        return np.stack([1-pred, pred], axis=1)
    
    def predict(self, x):
        return int(self.predict_proba(x)[0] > self.threshold)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
