# -*- coding: utf-8 -*-
"""
From Learning scikit-learn: Machine Learning in Python
"""

import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# Print info ttg dataset
print("Iris data set:", X_iris.shape, y_iris.shape)


# Pisahkan data menjadi training dan testing set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Get dataset with only the first two attributes (Sepal only)
X, y = X_iris[:, :2], y_iris

# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Print the training data
print("Training set:", X_train.shape, y_train.shape)
for i in range(len(X_train)):
    print(y_train[i], "=>", X_train[i])

# Draw graphics in 2 Dimension
colors = ['red', 'black', 'blue']
markers = ['o', 'x', '+']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i], marker=markers[i])

plt.title("IRIS Learning Set")
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


