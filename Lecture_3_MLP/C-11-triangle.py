# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 02:44:37 2018

@author: Mursito
"""

import sklearn as sk
import matplotlib.pyplot as plt
import triangle as s3

# Get dataset of triangles
X=[]
y=[]
for i in range(300):
    tangle, tside, tr = s3.triangle()
    tr.sort()
    X.append(tr)
    y.append(tangle)
    
legends=s3.side_types()
labels=s3.parameters()

    
# Pisahkan data menjadi training dan testing set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Print the training data
print("TRAINING DATA:")
for i in range(len(X_train)):
    print(y_train[i], "=>", scaler.inverse_transform(X_train[i]))

# Draw graphics in 3 Dimension
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = ['red', 'green', 'blue']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lj = len(y_train)    

for i in range(len(colors)):
    xs = [ X_train[j,0] for j in range(lj) if y_train[j]==i ]
    ys = [ X_train[j,1] for j in range(lj) if y_train[j]==i ]
    zs = [ X_train[j,2] for j in range(lj) if y_train[j]==i ]
    ax.scatter(xs, ys, zs, c=colors[i], marker='o')

ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_zlabel(labels[2])
plt.legend(legends)
plt.title("Triangle Learning Set")
plt.show()


