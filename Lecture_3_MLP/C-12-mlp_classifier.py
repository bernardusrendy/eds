# -*- coding: utf-8 -*-
"""
From Learning scikit-learn: Machine Learning in Python
MUST BE RUN AFTER C-11-triangle
"""

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), 
    activation='logistic', # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    solver='lbfgs',        # {‘lbfgs’, ‘sgd’, ‘adam’} 
    learning_rate='adaptive', # {‘constant’, ‘invscaling’, ‘adaptive’}
    max_iter=10000, 
    momentum=0.9,
    random_state=0)

mlp.fit(X_train,y_train)

y_mlp = mlp.predict(X_test)

print("TESTING:")
for i in range(len(X_test)):
    if (y_mlp[i] == y_test[i]):
        print(y_mlp[i], "==", scaler.inverse_transform(X_test[i]))
    else:
        print(y_mlp[i], "<>", scaler.inverse_transform(X_test[i]))
        
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_mlp))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lj = len(y_train)    

for i in range(len(colors)):
    xs = X_test[:, 0][y_mlp == i]
    ys = X_test[:, 1][y_mlp == i]
    zs = X_test[:, 2][y_mlp == i]
    ax.scatter(xs, ys, zs, c=colors[i], marker='+')

ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_zlabel(labels[2])
plt.legend(legends)
plt.title("Triangle MLP Testing")
plt.show()


