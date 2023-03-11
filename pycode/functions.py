#coding=UTF-8
#Éî¶ÈÑ§Ï°º¯Êý¿â
import sys,os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")
import numpy as np
from ch01.mnist import load_mnist
import pkl
import pickle

#sigmoid ¼¤»îº¯Êý£¬½«ÊýÖµ¹Ì¶¨ÔÚ0µ½1Ö®¼ä
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#ReLUÒ²ÊÇÒ»ÖÖ¼¤»îº¯Êý
def relu(x):
    return np.maximum(0,x)

#softmax·ÖÀàº¯Êý£¬Ò²¾ÍÊÇ°Ù·Ö±È£¬¼ø±ðÄÇ¸ö½á¹û¿ÉÄÜÐÔ×î´ó
def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

#ï¿½ï¿½mnistï¿½Å±ï¿½ï¿½Ð»ï¿½È¡ï¿½ï¿½ï¿½Ý¼ï¿½
def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test

#ï¿½ï¿½pklï¿½Ä¼ï¿½ï¿½ï¿½È¡ï¿½Ñ¾ï¿½Ñµï¿½ï¿½ï¿½Ãµï¿½È¨ï¿½Ø£ï¿½ï¿½ï¿½ï¿½Ò·ï¿½ï¿½ï¿½ï¿½Öµï¿½
def init_network():
    with open("./pkl/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
        return network
    

#ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
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



