# -*- coding: utf-8 -*-
"""
From Learning scikit-learn: Machine Learning in Python
MUST BE RUN AFTER C-01-dataset
"""

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,100),
    activation='logistic', 
    solver='lbfgs', 
    max_iter=10000)

mlp.fit(X_train,y_train)

y_mlp = mlp.predict(X_test)

print("Testing set:")
for i in range(len(X_test)):
    if (y_mlp[i] == y_test[i]):
        print(y_mlp[i], "==", X_test[i])
    else:
        print(y_mlp[i], "<>", X_test[i])
        
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_mlp))

for i in range(len(colors)):
    xs = X_test[:, 0][y_mlp == i]
    ys = X_test[:, 1][y_mlp == i]
    plt.scatter(xs, ys, c=colors[i], marker='+')


plt.title("IRIS MLP Testing")
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

