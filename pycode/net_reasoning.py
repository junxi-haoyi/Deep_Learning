#coding=UTF-8
import functions as fuc
import numpy as np


#单个图像处理
x, t = fuc.get_data()
network = fuc.init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = fuc.predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt+=1

#批处理图像
batch_size = 100
accuracy_batch = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = fuc.predict(network,x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_batch += np.sum(p == t[i:i+batch_size])




print(f"Accuracy_cnt:{str(float(accuracy_cnt/len(x)))}")
print(f"Accyracy of batch :{str(float(accuracy_batch/len(x)))}")








