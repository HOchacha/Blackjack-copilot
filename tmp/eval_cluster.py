import numpy as np
import random
import matplotlib.pyplot as plt
 
from itertools import combinations
 
np.random.seed(100)
num_data = 20
 
x1 = np.linspace(0.3,0.7,num_data)
error = np.random.normal(1,0.5,num_data)
x2 = 1.5*x1+2+error
 
X = np.stack([x1, x2], axis=1)

fig = plt.figure(figsize=(7,7))
fig.set_facecolor('white')
plt.scatter(x1, x2, color='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

def complete_diameter_distance(X):
    res = []
    for i, j in combinations(range(X.shape[0]),2):
        a_i = X[i, :]
        a_j = X[j, :]
        res.append(np.linalg.norm(a_i-a_j))
 
    return np.max(res)

print(complete_diameter_distance(X))
