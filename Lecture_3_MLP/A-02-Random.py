# -*- coding: utf-8 -*-
# Basic random, dictionary
# Various loops

# import namun masih pakai alias
import numpy.random as nr
import matplotlib.pyplot as mp

# inisialisasi list 10 elemen
data=[0]*30
print(data)

# Generate random values
print("Generate random values")
for i in range(len(data)):
    data[i] = nr.randint(1,4)*2
        
print("Data:", data)

data.sort()
print("Sorted:", data)

nr.shuffle(data)
print("Shuffled:", data)

# define dictionary for tabularization
tab={2:0, 4:0, 6:0}
print("Dictionary:", tab)

# tabularization, using dictionary
for e in data:
    tab[e] = tab[e] + 1
print("Tabular:", tab)

# extract tabulation info
lists = sorted(tab.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

mp.plot(x, y)
mp.show()






