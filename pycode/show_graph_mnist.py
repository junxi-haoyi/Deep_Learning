#coding=UTF-8
import pickle
from PIL import Image
import numpy as np
from ch01.mnist import load_mnist
import functions as fuc


#从pkl文件中读取已经训练好的权重，数据以字典的形式存放
#图片以灰度图像的形式呈现，单色通道便于信息的处理
#数组中的每一个元素代表一个像素点，数值大小代表白色程度，颜色越白，数值越大




def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

arry = np.array([5,3])
print(f"arry:{arry}")
(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)#(训练图像，训练标签)，（测试图像，测试标签）
#print(x_train.shape) (60000, 784)
#print(t_train.shape) (60000,)
#print(x_test.shape) (10000,784)
#print(t_test.shape) (10000)
img = x_train[0]
label = t_train[0]
print(label)
print(img)
print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)




