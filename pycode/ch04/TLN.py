import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from pycode.ch01.mnist import load_mnist


# sigmoid activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(x))


# softmax Classification functions
def softmax(x):
    # if ndim is 2 then need to calculate every dims softmax
    if x.ndim == 2:
        x = x.T
        x = x - x.max(axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x))


# cross entropy error
def cross_entropy_error(x, t):
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        t = t.reshape(1, t.size)

    if x.size == t.size:
        t = np.argmax(t, axis=1)

    return -np.sum(np.log(x[np.arange(x.shape[0]), t] + 1e-4))


# numerical gradient
def numerical_gradient(f, x):
    # minimum num avoid to
    h = 1e-4
    # array to store numerical gradient of every weight and bias
    grads = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])            
    while not it.finished:
        store_num = it[it.multi_index]
        it[it.multi_index] = store_num + 1e-4
        fxh1 = f(x) 




        

        













