import numpy as np

a = np.arange(5*5*3).reshape(5,5,3)
print(a)
print(np.transpose(a, (2,0,1)))