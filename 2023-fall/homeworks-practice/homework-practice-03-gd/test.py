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


# # YOUR CODE:
# from linear_regression import LinearRegression
# from descents import LossFunction

# descent_config = {
#     'descent_name': 'name',
#     'kwargs': {
#         'dimension': len(x_validate.columns),
#         'lambda_' : 0
#     }
# }

# for descent_name in ['full', 'stochastic', 'momentum', 'adam']:
#   descent_config['descent_name'] = descent_name
#   iterations = 0
#   min_err = np.inf
#   best_lr = 0
#   for lr in np.logspace(-5, -1,4):
#     descent_config['kwargs']['lambda_'] = lr
#     model = LinearRegression(
#       descent_config=descent_config,
#       max_iter=100000
#     ).fit(x_validate, y_validate)

#     curr_err = model.calc_loss(x_validate, y_validate)
#     if curr_err < min_err:
#       min_err = curr_err
#       best_lr = lr
#       iterations = model.epoch
#     # statistics.add((descent_name, model.epoch + 1, curr_err))
#     print("For {} gradient descent the best lambda is {} with MSE = {} for {} iterations".format(descent_name, best_lr, min_err, iterations))

batch_sizes = np.arange(5, 500, 10)
statistics = pd.DataFrame(columns=['batch_size', 'epochs', 'error', 'time'])

import time
# YOUR CODE:
descent_config = {
    'descent_name': 'stochastic',
    'kwargs': {
        'dimension': len(x_validate.columns),
        'batch_size': 0,
        'lambda_': 0.1
    }
}
for b_s in batch_sizes:
    descent_config['kwargs']['batch_size'] = b_s
    for k in range(10):
        start = time.time()
        model = LinearRegression(
            descent_config=descent_config,
            max_iter=10000
        ).fit(x_validate, y_validate)
        waited = time.time() - start

        curr_err = model.calc_loss(x_validate, y_validate)
        statistics.loc[len(statistics)] = [b_s, model.epoch, curr_err, waited]
        print(b_s, model.epoch, curr_err, waited)

display(statistics.groupby('batch_size').mean())