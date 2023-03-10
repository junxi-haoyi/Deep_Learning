#深度学习函数库
import numpy as np
from ch01.mnist import load_mnist
import pkl
import pickle

#sigmoid 函数，也称激活函数
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#ReLU函数 也是一种激活函数
def relu(x):
    return np.maximum(0,x)

#softmax函数，将数据进行分类
def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

#从mnist脚本中获取数据集
def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test

#从pkl文件读取已经训练好的权重，并且返还字典
def init_network():
    with open("./pkl/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
        return network
    

#三级的神经网络推理运算
def predict(network,x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z3,W3) + b3
    y = sigmoid(a3)

    return y



