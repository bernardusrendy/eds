# -*- coding: utf-8 -*-
# Basic random, dictionary
# Various loops

import numpy as np

# inisialisasi array (vektor 1x3)
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4., 5., 6.])
v2t = v2.transpose()

print("v1=", v1)
print("v2=", v2)
print("v2t=", v2t)

print("size v1", v1.size)
print("size v2", v2.size)
print("size v2t", v2t.size)

print("v1+v2=",v1+v2)
print("v1*v2=",v1*v2)
print("v1.v2=", np.dot(v1, v2))

print("v1*v2t=",v1*v2t)

