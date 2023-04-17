import numpy as np


x = np.zeros((3,3,3))
x[1,2][1:] = 1 
print(x)
print(x[1,2])