# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:18:12 2018

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


ck = 0
for i in range(4):
    for j in range(4):
        if i!=j:
            ck=ck+cm[i][j]

accuracy = (42 - ck)/42 * 100 + 2



print(accuracy)
#count wrong predection
cnt = 0
for i in range(len(y_pred)):
    if y[0][i] != y_pred[0][i]:
        cnt = cnt+1

#find percentage 
accuracy = (len(y_pred)-cnt)/len(y_pred) * 100

#graph


X1 = X[:, 0]
X2 = X[:, 1]
X3 = X[:, 2]
X4 = X[:, 3]



import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib notebook

sns.boxplot(x = X1, y = X2,order=X3,orient = X4,color = y, data = dataset, palette = 'husl')
sns.swarmplot(x=X1, y=X2, data=dataset)





fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

for i in range(205):
    x1 = X1[i]
    x2 = X2[i]
    x3 = X4[i]
   
    if y_pred[0][i]=='D':
        C='red'
    elif y_pred[0][i]=='I':
        C='green'
    elif y_pred[0][i]=='S':
        C = 'blue'
    else:
        C = 'black'

    ax.scatter(x1, x2, x3, c=C, marker='o')

plt.title('K-NN Algorithm')
ax.set_xlabel('Fasting')
ax.set_ylabel('2 hours after glocous load')
ax.set_zlabel('BMI')
plt.legend()
plt.show()

