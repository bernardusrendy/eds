# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 02:44:37 2018

@author: Mursito
"""
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

index_max = len(iris_data)

# menyatukan label dan data
table = []
for i in range(index_max):
    table.append([iris_labels[i], iris_data[i]])

np.random.shuffle(table)

# ambil learn set 80% pertama
index_learn = index_max * 8 // 10
learn_set=table[0:index_learn]

# sisanya testing test 30% terakhir
index_test = index_max * 7 // 10
test_set=table[index_test:index_max-1]


print("LEARNING SET:")
learn_count=[0,0,0]
for i in range(len(learn_set)):
    print(learn_set[i][0], ":" , learn_set[i][1])
    learn_count[learn_set[i][0]] += 1

# Gambar grafiknya dalam 3 dimensi
# data[2] dan data[3] dijumlah agar dapat tampil 3D
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
colours = ("r", "b")
X = []
for iclass in range(3):
    X.append([[], [], []])
    for i in range(len(learn_set)):
        if learn_set[i][0] == iclass:
            X[iclass][0].append(learn_set[i][1][0])
            X[iclass][1].append(learn_set[i][1][1])
            X[iclass][2].append(sum(learn_set[i][1][2:]))
colours = ("r", "g", "y")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(3):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()

