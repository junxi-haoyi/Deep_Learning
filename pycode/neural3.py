import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))



def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['B1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([0.1,0.2],[0.2,0.3],[0.4,0.5])
    network['B2']=np.array([0.1,0.2])
    network['W3']=np.arrary([[0.1,0.3],[]])
                           
