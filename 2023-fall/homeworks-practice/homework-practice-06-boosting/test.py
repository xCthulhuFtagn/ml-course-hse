from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


sns.set(style='darkgrid')
filterwarnings('ignore')

x = load_npz('x.npz')
y = np.load('y.npy')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1337)

x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=1337)

from boosting import Boosting
from sklearn.model_selection import KFold, cross_val_score
kf = KFold()
def gb_mse_cv(params, cv=kf, X=x_train, y=y_train):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']), 
              'base_model_params': dict(params['base_model_params']), 
             'learning_rate': params['learning_rate'],
             'early_stopping_rounds': int(params['early_stopping_rounds']),
             'subsample': params['subsample']
    }
    
    # we use this params to create a new LGBM Regressor
    model = Boosting(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X=X, y=y, cv=cv, scoring="roc_auc", n_jobs=-1, error_score='raise').mean()

    return score


# YOUR CODE:
from hyperopt import fmin, tpe, hp, anneal, Trials

space = {
    'base_model_params': {'max_depth' : hp.quniform('max_depth', 2, 20, 1)},
    'n_estimators': hp.quniform('n_estimators', 10, 2000, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'early_stopping_rounds': hp.quniform('early_stopping_rounds', 2, 10, 1),
    'subsample' : hp.choice('subsample', [i/10 for i in range(1, 11, 1)])
}

trials = Trials()

best=fmin(fn=gb_mse_cv, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=50, # maximum number of iterations
          trials=trials, # logging
         )
print(best)
boosting = Boosting(base_model_params={'max_depth' : best['max_depth']}, 
                    n_estimators=int(best['n_estimators']),
                    learning_rate=best['learning_rate'],
                    early_stopping_rounds=int(best['early_stopping_rounds']),
                    subsample=int(best['subsample']))

print(best)
print(roc_auc_score(y_test, boosting.predict(x_test)))