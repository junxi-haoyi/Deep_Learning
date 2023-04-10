import sys
sys.path.append(".")
import pycode.functions as fc
import numpy as np




#Init layer's weights and bias
class TwoLaterNet:
    # def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
    #     self.params={}
    #     self.params['w1'] = np.random.normal(loc=0.5,scale=0.1,size=(input_size,hidden_size)) * weight_init_std
    #     self.params['b1'] = np.zeros(hidden_size)
    #     self.params['w2'] = np.random.normal(loc=0.5,scale=0.1,size=(hidden_size,output_size)) * weight_init_std
    #     self.params['b2'] = np.zeros(output_size)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self,x):
        w1, w2 = self.params['w1'],self.params['w2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,w1) + b1
        a1 = fc.sigmoid(a1)
        a2 = np.dot(a1,w2) + b2
        y = fc.softmax(a2)

        return y
    
    def loss(self,x,t):
        y = self.predict(x)

        return fc.cross_entropy_error_one_hot(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        return np.sum(y==t) / float(x.shape[0])
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        grads['w1'] = fc.numerical_gradient(loss_W,self.params['w1'])
        grads['b1'] = fc.numerical_gradient(loss_W,self.params['b1'])
        grads['w2'] = fc.numerical_gradient(loss_W,self.params['w2'])
        grads['b2'] = fc.numerical_gradient(loss_W,self.params['b2'])

        return grads



    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, w1) + b1
        z1 = fc.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = fc.softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, w2.T)
        dz1 = fc.sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads



