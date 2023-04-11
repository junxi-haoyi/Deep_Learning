import torch
import matplotlib.pyplot as plt

x = torch.normal(0.5, 0.1, size=(5,3))
y = torch.normal(0.5, 0.1, size=(3,5))
z = torch.mm(x, y)
print(z)

fig = plt.figure()
axe = plt.axes()

axe.hist(z, bins=4, edgecolor='r')

plt.show()
