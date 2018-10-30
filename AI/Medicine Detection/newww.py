# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:05:56 2018

@author: karigor
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('data3.csv')
X = dataset.iloc[:, :-1].values
M = pd.DataFrame(X).values
y1 = dataset.iloc[:, 4].values
N = pd.DataFrame(y1).values


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


#Logictic Function
def CallFunction(X, y):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X, y)

    # Predicting the Test set results
    y_pred = classifier.predict_proba(X)
    return y_pred


#probabily for each catagory
def Pro(nub, X, y1):
    
    y = []

    for i in range(len(y1)):
        y.append(y1[i][nub])
    
    y_pred = CallFunction(X, y)

    y_pro = []

    for i in range(len(y1)):
        y_pro.append(y_pred[i][1]) 

    return y_pro


y_pred0 = Pro(0, X, y1)
y_pred1 = Pro(1, X, y1)
y_pred2 = Pro(2, X, y1)
y_pred3 = Pro(3, X, y1)



Y_pred = np.zeros([len(y1), 4])
cnt=0
for i in range(len(y1)):
    mx = 0
    mx = max(y_pred0[i], mx)
    mx = max(y_pred1[i], mx)
    mx = max(y_pred2[i], mx)
    mx = max(y_pred3[i], mx)
    
    if mx==y_pred0[i]:
        Y_pred[i][0] = 1
        if y1[i][0] != 1:
            cnt=cnt+1
    elif mx==y_pred1[i]:
        Y_pred[i][1] = 1
        if y1[i][1] != 1:
            cnt=cnt+1
    elif mx==y_pred2[i]:
        Y_pred[i][2] = 1
        if y1[i][2] != 1:
            cnt=cnt+1
    elif mx==y_pred3[i]:
        Y_pred[i][3] = 1
        if y1[i][3] != 1:
            cnt=cnt+1

accuracy = (len(y1)-cnt)/len(y1)

Y1 = Y_pred[:, 0]
Y2 = Y_pred[:, 1]
Y3 = Y_pred[:, 2]
Y4 = Y_pred[:, 3]

X1 = X[:, 0]
X2 = X[:, 1]
X3 = X[:, 2]
X4 = X[:, 3]





fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')

for i in range(len(y1)):
    x1 = X1[i]
    x2 =X2[i]
    x3 = X3[i]
   
    if N[i]=='D':
        C='red'
    elif N[i]=='I':
        C='green'
    elif N[i]=='S':
        C = 'blue'
    else:
        C = 'black'

    ax.scatter(x1, x2, x3, c=C, marker='o')

plt.show()


