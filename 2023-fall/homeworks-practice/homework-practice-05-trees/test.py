import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import Colormap, ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style='whitegrid')

### ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from random import randint
from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score

mushrooms = pd.read_csv('agaricus-lepiota.data')
# label_encoder = LabelEncoder()
# for column in mushrooms.columns:
#     mushrooms[column] = label_encoder.fit_transform(mushrooms[column])

# X = mushrooms.columns[1:]
# mask = np.repeat([True,False], len(mushrooms)//2)
# if len(mask) < len(mushrooms): mask = np.append(mask, randint(0, 1))
# np.random.shuffle(mask)
# mask = mask.astype(bool)
# X_train, Y_train = mushrooms[mask][X], mushrooms[mask][mushrooms.columns[0]]
# X_test, Y_test = mushrooms[~mask][X], mushrooms[~mask][mushrooms.columns[0]]

from hw5code import DecisionTree
# feature_types = ['categorical']*X_test.shape[1]
# tree = DecisionTree(feature_types=feature_types).fit(X_train, Y_train)

# Y_p = tree.predict(X_test)
# print(accuracy_score(Y_test, Y_p))

mush_y = pd.get_dummies(mushrooms.iloc[:, 0], drop_first=True).astype(int)
mush_X = mushrooms.iloc[:, 1:]
OHE_mush_X = pd.get_dummies(mush_X)

f_types = ['categorical']*mush_X.shape[1]
ans = cross_val_score(estimator=DecisionTree(feature_types=f_types),scoring=make_scorer(accuracy_score), X=mush_X, y=mush_y, cv=10, error_score='raise')