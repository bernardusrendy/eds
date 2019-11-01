# -*- coding: utf-8 -*-
"""
From Learning scikit-learn: Machine Learning in Python
MUST BE RUN AFTER C-01-dataset
"""

# buat data latih binary
import copy

y1_train = copy.deepcopy(y_train)
y1_test = copy.deepcopy(y_test)
legends = np.append(iris.target_names, 'others')

# ubah semua yang bukan chosen jadi 3
chosen = 1
for i in range(len(y1_train)) :
    if (y1_train[i] != chosen):
        y1_train[i] = 3

for i in range(len(y1_test)) :
    if (y1_test[i] != chosen):
        y1_test[i] = 3

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10),
    activation='tanh', 
    solver='adam', 
    max_iter=10000
    )

# Latihan, mamakai data latih
mlp.fit(X_train,y1_train)

# Uni memakai data uji
y_mlp = mlp.predict(X_test)

print("Testing set:")
for i in range(len(X_test)):
    if (y_mlp[i] == y1_test[i]):
        print(y_mlp[i], "==", X_test[i])
    else:
        print(y_mlp[i], "<>", X_test[i])
        
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y1_test,y_mlp))

colors = ['red', 'greenyellow', 'blue', 'orange']
for i in range(len(colors)):
    xs = X_test[:, 0][y_mlp == i]
    ys = X_test[:, 1][y_mlp == i]
    plt.scatter(xs, ys, c=colors[i], marker='+')

plt.title("IRIS MLP Testing")
plt.legend(legends)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

