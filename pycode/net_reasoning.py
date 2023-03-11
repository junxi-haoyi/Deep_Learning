import functions as fuc
import numpy as np



x, t = fuc.get_data()
network = fuc.init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = fuc.predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt+=1

print(f"Accuracy_cnt:{str(float(accuracy_cnt/len(x)))}")







