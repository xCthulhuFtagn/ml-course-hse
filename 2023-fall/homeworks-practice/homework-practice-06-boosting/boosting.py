from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
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

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        # bootstrap
        # https://stackoverflow.com/questions/54058718/why-random-sample-cant-handle-numpy-arrays-but-random-choices-can
        rng = np.random.default_rng()
        mask = np.random.choice([False, True], x.shape[0], p=[0.85, 0.15])
        new_model = DecisionTreeRegressor().fit(x[mask, :], (y - predictions)[mask])
        self.gammas.append(self.find_optimal_gamma(y, predictions, new_model.predict(x)))
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
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
                    if (self.validation_loss - val_loss).sum() == 0: break
                    else: self.validation_loss = val_loss
        if self.plot: pass
            
    def predict_proba(self, x):
        pred = 0
        for gamma, model in zip(self.gammas, self.models):
            pred += gamma * model.predict(x)
        pred = self.sigmoid(pred)
        return np.stack([pred, 1- pred], axis=1)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
