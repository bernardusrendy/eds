# -*- coding: utf-8 -*-
# Basic List
# Various loops

skalar=1

data=['A','B','C',"Kalimat", 5, 1.23]


# Using while
print("While index")
i=0
while i<len(data):
    print(data[i])
    i=i+1
    

print("For index") 
for i in range(len(data)):
    print(i)
    print(data[i])

print("For element", end='\n')
for e in data:
    print(e)

