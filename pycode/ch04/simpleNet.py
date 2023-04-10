
import os,sys
sys.path.append(".")
import pycode.functions as fc
import numpy as np

net = fc.simple_net()
#print(net.w)

x = np.array([0.6,0.9],dtype=float)
print(x)
p = net.predict(x)

t = np.array([0,0,1],dtype=float)
a = net.loss(x, t)
print(a)

def f(W):
    return net.loss(x, t)

dw = fc.numerical_gradient(f, net.w)
print(dw)


