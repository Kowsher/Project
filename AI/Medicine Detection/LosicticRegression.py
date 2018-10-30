# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:05:56 2018

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

#Logictic Function
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
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