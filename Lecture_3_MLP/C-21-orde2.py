# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 02:44:37 2018

@author: Mursito
"""

import sklearn as sk
import matplotlib.pyplot as plt
import orde2 as orde2

# Get dataset of triangles
X=[]
y=[]
for i in range(100):
    rtype, t, signal = orde2.step()
    X.append(signal)
    y.append(rtype)
    
legends=orde2.respond_types()
labels=orde2.parameters()
    
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
print("TRAINING SET:")
ctype=[0,0,0,0]
for i in range(len(y_train)):
    ctype[y_train[i]] += 1

for i in range(len(ctype)):
    print(legends[i], "=", ctype[i])

# Draw graphics in 3 Dimension
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = ['red', 'green', 'blue', "black"]

lenx = len(X_train[0])    
ys = [0]*lenx
xs = []
xs.extend(range(lenx))

for i in range(len(X)):
    zs = X[i]
    plt.plot(xs, zs, c=colors[y[i]])

plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.title("Orde2 Learning Set")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lenx = len(X_train[0])    

ys = [0]*lenx
xs = []
xs.extend(range(lenx))

for i in range(len(X_train)):
    zs = X_train[i]
    ys = [e+1 for e in ys]
    ax.plot(xs, ys, zs, c=colors[y_train[i]])

ax.set_ylabel("sample")
ax.set_xlabel(labels[0])
ax.set_zlabel(labels[1])
#plt.legend(legends)
plt.title("Orde2 Learning Set")
plt.show()


