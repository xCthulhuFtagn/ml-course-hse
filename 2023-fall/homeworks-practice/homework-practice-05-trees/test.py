import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import Colormap, ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

students_df = pd.read_csv('/home/owner/Documents/DEV/ML/ml-course-hse/2023-fall/homeworks-practice/homework-practice-05-trees/students.csv')

from hw5code import find_best_split
best_feature = None
best_score = 0
statistics = pd.DataFrame(columns = ['threshold', 'gini', 'feature'])
for feature, i in zip(students_df.columns[:-1], range(len(students_df.columns) - 1)):
    thresholds, ginis, threshold_best, gini_best = find_best_split(students_df[feature], students_df[students_df.columns[-1]])
    statistics = pd.concat([statistics, pd.DataFrame({'threshold' : thresholds, 'gini': ginis, 'feature': feature})], ignore_index = True)

fig = sns.lineplot(data = statistics, x = 'threshold', y = 'gini', hue = 'feature').get_figure()
fig.savefig(f"fig.png")


# x = [1, 2, 3, 4, 5, 6, 99]
# y = [0, 1, 1, 0, 0, 0, 1]

# thresholds, ginis, threshold_best, gini_best = find_best_split(x, y)

# print(thresholds)
# print(ginis)