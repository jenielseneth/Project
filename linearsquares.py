import numpy as np
from numpy.linalg import inv

def get_A_B_C(data, tc, omega, phi, m):
    f = np.zeros(len(data))
    g = np.zeros(len(data))
    for i in range(len(data)):
        f[i] = -np.power((tc-i), m)
        g[i] = np.power((tc-i), m)*np.cos(omega*np.log(tc-i)+phi)
    X = np.vstack([np.ones(len(data)), f, g]).T
    xtx = np.matmul(X.T, X)
    xty = X.T.dot(data)
    b = np.matmul(inv(xtx), xty)
    return b