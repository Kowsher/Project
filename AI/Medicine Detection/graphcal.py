import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X, 
                      y=y,
                      clf=classifier, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel(X.columns[0], size=14)
plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)