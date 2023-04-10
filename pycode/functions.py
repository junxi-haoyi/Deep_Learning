#coding=UTF-8
import os,sys
sys.path.append("./DL/Deep_Learning")
import sys,os
sys.path.append(".")
import numpy as np
from ch01.mnist import load_mnist
#import pkl
import pickle


# sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))


# ReLU
def relu(x):
    return np.maximum(0,x)


# softmax
def softmax(x):
    # c = np.max(a)
    # exp_a=np.exp(a-c)
    # sum_exp_a=np.sum(exp_a)
    # y=exp_a/sum_exp_a
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# mnist
def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test

#pkl
def init_network():
    with open("./pkl/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
        return network


#
def predict(network,x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = sigmoid(a3)

    return y


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


#
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


# mini-batch?? 

# one-hot
def cross_entropy_error_one_hot(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    xsize = y.shape[0]
    return -np.sum(np.log(y[np.arange(xsize), t] + 1e-7)) / xsize


# one-hot
def cross_entropy_error_none_one_hot(y, t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    xsize = y.shape[0]


# def numerical_gradient(f,x):
#     h = 1e-4
#     grad = np.zeros_like(x)

#     it = np.nditer(x,flags=["multi_index"],op_flags=['readwrite'])
#     while not it.finished:
#         str = x[it.multi_index]
#         x[it.multi_index] = str + h
#         fh1 = f(x)

#         x[it.multi_index] = str - h
#         fh2 = f(x)

#         grad[it.multi_index] = (fh1 - fh2) / (2*h)
#         x[it.multi_index] = str
#         it.iternext()

#     return grad
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # ��ԭֵ
        it.iternext()

    return grad





#
class simple_net:
    def __init__(self):
        self.w = np.random.normal(loc=0.5,scale=0.1)
        self.w = np.clip(self.w,0.0,1.0)

    def predict(self,x):
        return np.dot(x,self.w)

    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error_one_hot(y, t)

        return loss



def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)






