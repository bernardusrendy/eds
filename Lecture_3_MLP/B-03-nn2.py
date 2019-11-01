# -*- coding: utf-8 -*-

# Multi Layer Neural Network
# Supervised learning

import numpy as np

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# adjustment function
def adjust(x):
    return x*(1-x)

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,0],
                [1,1,0] ])
    
# supervised output dataset            
y = np.array([[0,1,0,1]]).T

# randomly initialize our weights with mean 0
np.random.seed(1)
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

l1 = sigmoid(np.dot(X,syn0))
l2 = sigmoid(np.dot(l1,syn1))

print("INITIAL")
print("Output :")
print(l2)

for j in range(60000):

 	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))

    # calculate error layer2
    l2_error = y - l2    
    if (j% 10000) == 0:
        print("Error:",str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*adjust(l2)

    # calculate error layer 1
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * adjust(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


print("AFTER TRAINING")
print("Output :")
print(l2)

# Action process
print("INTO ACTIONS:")
print("Trained Inputs:")

x1 = [0,1,1]
l1 = sigmoid(np.dot(x1, syn0))
l2 = sigmoid(np.dot(l1,syn1))
print(x1, " = ", l2)

x1 = [0,0,1]
l1 = sigmoid(np.dot(x1, syn0))
l2 = sigmoid(np.dot(l1,syn1))
print(x1, " = ", l2)

print("Untrained Inputs:")
x1 = [1,0,1]
l1 = sigmoid(np.dot(x1, syn0))
l2 = sigmoid(np.dot(l1,syn1))
print(x1, " = ", l2)

x1 = [0,1,0]
l1 = sigmoid(np.dot(x1, syn0))
l2 = sigmoid(np.dot(l1,syn1))
print(x1, " = ", l2)



