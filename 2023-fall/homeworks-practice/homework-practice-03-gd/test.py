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

# YOUR CODE:
from linear_regression import LinearRegression
import descents

descent_config = {
    'descent_name': 'name',
    'kwargs': {
        'dimension': len(x_validate.columns)
    }
}

statisticsVanilla = pd.DataFrame(columns=['descent', 'R^2', 'error', 'lr'])
statisticsStochastic = pd.DataFrame(columns=['descent', 'R^2', 'error', 'lr'])
statisticsMomentum = pd.DataFrame(columns=['descent', 'R^2', 'error', 'lr'])
statisticsAdam = pd.DataFrame(columns=['descent', 'R^2', 'error', 'lr'])


for descent_name in ['full', 'stochastic', 'momentum', 'adam']:
  descent_config['descent_name'] = descent_name
  iterations = 0
  min_err = np.inf
  best_lr = 0
  for lr in np.logspace(-5, -1,4):
    descent_config['kwargs']['lambda_'] = lr
    descent_config['regularized'] = True
    
    for mu in np.logspace(-5, -1, 4):
      descent_config['kwargs']['mu'] = mu
      model = LinearRegression(
          descent_config=descent_config,
          max_iter=10000
      ).fit(x_validate, y_validate)

      curr_err = model.calc_loss(x_validate, y_validate)
      y_p = model.predict(x_test)
      R_2 = descents.R_square(y_test, y_p)
      
      
      if isinstance(model.descent, descents.StochasticDescent):
        statisticsStochastic.loc[len(statisticsStochastic)] = [model.descent.__class__, R_2, curr_err, lr]
      elif isinstance(model.descent, descents.MomentumDescent):
        statisticsMomentum.loc[len(statisticsMomentum)] = [model.descent.__class__, R_2, curr_err, lr]
      elif isinstance(model.descent, descents.Adam):
        statisticsAdam.loc[len(statisticsAdam)] = [model.descent.__class__, R_2, curr_err, lr]
      elif isinstance(model.descent, descents.VanillaGradientDescent):
        statisticsVanilla.loc[len(statisticsVanilla)] = [model.descent.__class__, R_2, curr_err, lr]
      
    descent_config['regularized'] = False
    descent_config['kwargs'].pop('mu')
    
    model = LinearRegression(
        descent_config=descent_config,
        max_iter=10000
    ).fit(x_validate, y_validate)
    
    curr_err = model.calc_loss(x_validate, y_validate)
    y_p = model.predict(x_test)
    R_2 = descents.R_square(y_test, y_p)
    
    if isinstance(model.descent, descents.StochasticDescent):
      statisticsStochastic.loc[len(statisticsStochastic)] = [model.descent.__class__, R_2, curr_err, lr]
    elif isinstance(model.descent, descents.MomentumDescent):
      statisticsMomentum.loc[len(statisticsMomentum)] = [model.descent.__class__, R_2, curr_err, lr]
    elif isinstance(model.descent, descents.Adam):
      statisticsAdam.loc[len(statisticsAdam)] = [model.descent.__class__, R_2, curr_err, lr]
    elif isinstance(model.descent, descents.VanillaGradientDescent):
      statisticsVanilla.loc[len(statisticsVanilla)] = [model.descent.__class__, R_2, curr_err, lr]
    
