import os
import numpy as np
from datagen import *
from utils import *
import cv2

# import matplotlib.pyplot as plt

train_ids = next(os.walk('dataset'))[1]

val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

image_size = 256
train_path = "dataset/"
batch_size = 16

valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
    
model = UNet()
model.load_weights('unet-1583082834.h5')
model.summary()

x, y = valid_gen.__getitem__(1)
result = model.predict(x)
result = result > 0.5

cv2.imwrite('result/original.png', x[0]*255)
cv2.imwrite('result/mask.png', result[0]*255)

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")

# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")