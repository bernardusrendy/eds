# -*- coding: utf-8 -*-
# Basic random, dictionary
# Various loops

import numpy as np

# inisialisasi list 10 elemen
a = np.array([1., 2., 3.])
v2 = np.array([4., 5., 6.])
v2t = v2.transpose()

print("v1=", v1)
print("v2=", v2)
print("v2t=", v2t)

print("v1+v2=",v1+v2)
print("v1*v2=",v1*v2)
print("v1.v2=",np.dot(v1, v2))

print("v1*v2t=",v1*v2t)

