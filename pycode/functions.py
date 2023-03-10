#深度学习函数库
import numpy as np

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