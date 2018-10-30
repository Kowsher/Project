# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:42:55 2018

@author: karigor
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset = pd.read_csv('data5.csv')
X = dataset.iloc[:, :-1].values
M = pd.DataFrame(X)
y = dataset.iloc[:, 4].values
y = N = pd.DataFrame(y)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#treaining model
X_train = pd.DataFrame(X)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)

tree_clf = DecisionTreeClassifier(random_state=42)

# Grow a Decision Tree
tree_clf.fit(X, y)

from numpy import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plot_feature_space(classifier, X, y, axes=[-16, 15, -20, 20])