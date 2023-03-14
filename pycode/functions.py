#coding=UTF-8
#深度学习函数库
import sys,os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")
import numpy as np
from ch01.mnist import load_mnist
import pkl
import pickle

#sigmoid 激活函数，将数值固定在0到1之间
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#ReLU也是一种激活函数
def relu(x):
    return np.maximum(0,x)

#softmax分类函数，也就是百分比，鉴别那个结果可能性最大
def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

#读取mnist图像数据集
def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test

#初始化神经网络，从pkl文件读取权重，以字典的形式返回
def init_network():
    with open("./pkl/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
        return network
    

#进行神经网络推理运算
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




#损失函数中的均方误差，检测数据的有效性，数据的损失程度
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


#损失函数中的交叉熵误差
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


#mini-batch学习 选取批量计算损失误差











