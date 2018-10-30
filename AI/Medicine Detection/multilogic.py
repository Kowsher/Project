# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:26:59 2018

@author: karigor
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:44:28 2018

@author: karigor
"""
# Importing the libraries
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


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y)

#predection 
y_pred = classifier.predict(X_train)
y_pred = pd.DataFrame(y_pred)

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


