# -*- coding: utf-8 -*-

# Basic Neural Network
# Supervised learning

import numpy as np
import numpy.random as nr


# sigmoid function
#float sigmoid(float x) {
#    return ..        
#}

def sigmoid(x):    
    return 1/(1 + np.exp(-x))

# adjustment function
def adjust(x):
    return x*(1-x)

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,0],
                [1,1,0] ])
    
# supervised output dataset            
y0 = np.array([[0,1,0,1]]).T

# initialize weights randomly with mean 0
np.random.seed(1)
syn0 = 2*nr.random((3,1)) - 1

# aksi hitung neural network
ys = np.dot(X,syn0)
y = sigmoid(ys)

print("INITIAL")
print("Synapsis: ")
print(syn0)
print("Output :")
print(y)

# Learning process
for iter in range(10):
    
    # input
    xx = X
    
    # forward propagation
    y = sigmoid(np.dot(xx,syn0))

    # calculate error
    y_error = y0 - y

    # calculate adjustment
    delta = y_error * adjust(y)

    # update weights
    syn0 = syn0 + np.dot(xx.T,delta)

print("AFTER TRAINING")
print("Synapsis: ")
print(syn0)
print("Output :")
print(y)

# Action process
print("INTO ACTIONS:")
print("Trained Inputs:")

x1 = [0,1,1]
y1 = sigmoid(np.dot(x1, syn0))
print(x1, " = ", y1)

x1 = [0,0,1]
y1 = sigmoid(np.dot(x1, syn0))
print(x1, " = ", y1)


print("Untrained Inputs:")
x1 = [1,0,1]
y1 = sigmoid(np.dot(x1, syn0))
print(x1, " = ", y1)

x1 = [0,1,0]
y1 = sigmoid(np.dot(x1, syn0))
print(x1, " = ", y1)



