from PIL import Image
import numpy as np
from ch01.mnist import load_mnist
import sys
sys.path.append(".")


def img_show(im):
    pil_img = Image.fromarray(np.uint8(im))
    pil_img.show()


Array = np.array([5, 3])
print(f"Array:{Array}")
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#print(x_train.shape) (60000, 784)
#print(t_train.shape) (60000,)
#print(x_test.shape) (10000,784)
#print(t_test.shape) (10000)
img = x_train[0]
label = t_train[0]
print(label)
print(img)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)




