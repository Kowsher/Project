# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:58:13 2018

@author: karigor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('data3.csv')
X = dataset.iloc[:, :-1].values
M = pd.DataFrame(X)
y1 = dataset.iloc[:, 4].values
N = pd.DataFrame(y1)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y1 = labelencoder.fit_transform(y1)
y1 = y1.reshape(len(y1),1)
onehotencoder = OneHotEncoder(categorical_features = [0])
y1 = onehotencoder.fit_transform(y1).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



y_train = []
for i in range(len(y1)):
    y_train.append(y1[i][2])





# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y_train)

# Predicting the Test set results
# Predicting the Test set results
y_pred = classifier.predict(X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

# Visualising the Training set results

X_train = X
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train






X1, X2, X3, X4 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01), sparse=True)



X1, X2, X3, X4=X_set[:, 0],X_set[:, 1],X_set[:, 2],X_set[:, 3]
kk = classifier.predict(np.array([X1.ravel(), X2.ravel(),X3.ravel(), X2.ravel()]).T).reshape(X1.shape)



plt.contourf(X1, X2, kk)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()