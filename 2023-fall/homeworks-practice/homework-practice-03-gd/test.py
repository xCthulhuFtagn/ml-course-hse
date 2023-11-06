import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from descents import get_descent
from linear_regression import LinearRegression

data = pd.read_csv('/home/owner/Desktop/Documents/DEV/ML/ml-course-hse/2023-fall/homeworks-practice/homework-practice-03-gd/autos.csv')

data = data[(data['kilometer']  > 2000) & (data['autoAgeMonths'] < 300)]

categorical = ['brand', 'model', 'vehicleType', 'gearbox', 'fuelType', 'notRepairedDamage']
#  bool values as categorical because it will give the same result
numeric = ['powerPS', 'kilometer', 'autoAgeMonths']
other = []

data['bias'] = 1
other += ['bias']

x = data[categorical + numeric + other]
y = np.log1p(data['price'])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# only works with sparse = False
column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('scaling', StandardScaler(), numeric),
    ('other',  'passthrough', other)
]).set_output(transform='pandas')

x = column_transformer.fit_transform(x)

# YOUR CODE (data split into train/val/test):
# from sklearn.cross_decomposition import train_test
# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
x_train, x_validate, x_test = np.split(x.sample(frac=1, random_state=42), [int(.8*len(x)), int(.9*len(x))])
y_train, y_validate, y_test = np.split(y.sample(frac=1, random_state=42), [int(.8*len(y)), int(.9*len(y))])

statistics = pd.DataFrame(columns=['descent', 'iteration', 'error'])

# YOUR CODE:
from linear_regression import LinearRegression
from descents import LossFunction
from descents import R_square

descent_config = {
    'descent_name': 'name',
    'kwargs': {
        'dimension': len(x_validate.columns),
    }
}

statistics = pd.DataFrame(columns=['descent', 'LossFunction', 'iteration', 'error'])

for descent_name in ['full', 'stochastic', 'momentum', 'adam']:
  descent_config['descent_name'] = descent_name
  for l_f in [LossFunction.MAE, LossFunction.Huber]:
    descent_config['kwargs']['loss_function'] = l_f
    iterations = 0
    min_err = np.inf
    best_lr = 0
    best_R_2 = 0
    for lr in np.logspace(-5, -1,4):
        descent_config['kwargs']['lambda_'] = lr
        model = LinearRegression(
        descent_config=descent_config,
        max_iter=100000
        ).fit(x_validate, y_validate)

        curr_err = model.calc_loss(x_validate, y_validate)
        if curr_err < min_err:
            min_err = curr_err
            best_lr = lr
            iterations = model.epoch
            best_R_2 = R_square(y_test, model.predict(x_test))
        statistics.loc[len(statistics)] = [descent_name,l_f, model.epoch, curr_err]
    print("For {} gradient descent the best lambda is {} with {} = {} for {} iterations and R^2 = {}".format(descent_name, best_lr, l_f, min_err, iterations, best_R_2))