## Imports
import os
import sys
import random
import calendar
import time

import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import zipfile

from datagen import *
from utils import *

## Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

#### Step 1
## Download Dataset form Google Drive
## local_download_path = os.path.expanduser('~/Workspace/')
## 1. Authenticate and create the PyDrive client.
gauth = GoogleAuth()
## gauth.LocalWebserverAuth()
gauth.CommandLineAuth()

drive = GoogleDrive(gauth)

## choose a local (colab) directory to store the data.
local_download_path = os.path.abspath('tmp')
try:
  os.makedirs(local_download_path)
except: pass

## 2. Auto-iterate using the query syntax
##    https://developers.google.com/drive/v2/web/search-parameters
gfilename = 'dataset_400.zip'

print ("=> Start")
# if os.path.isfile(os.path.join(local_download_path, gfilename)):
#     print ("=> File exist: continue")
# else:
    print ("=> File not exist: start download file")
    file_list = drive.ListFile({'q': "title='" + gfilename + "'"}).GetList()
    for gFiles in file_list:
      # 3. Create & download by id.
      if gFiles['title'] == gfilename:
        print('title: %s, id: %s' % (gFiles['title'], gFiles['id']))
        fname = os.path.join(local_download_path, gFiles['title'])
        print('=> Downloading to {}'.format(fname))
        f_ = drive.CreateFile({'id': gFiles['id']})
        f_.GetContentFile(fname)
        print ("=> Download successes")
  

#### Step 2
## Extract Zip
print ("=> Start extract zip file")
local_extract_path = os.path.abspath('')
with zipfile.ZipFile(os.path.join(local_download_path, gfilename), 'r') as zip_ref:
    zip_ref.extractall(local_extract_path)

#### Step 4
## Hyperparameters
image_size = 256
train_path = "dataset/"
epochs = 1000
batch_size = 16

## Training Ids
train_ids = next(os.walk(train_path))[1]

## Validation Data Size
val_data_size = 499

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

#### Step 5
## Excute Dataset preparing function
gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)

#### Step 6
## Optional run
# r = random.randint(0, len(x)-1)
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(x[r])
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(np.reshape(y[r], (image_size, image_size)), cmap="gray")

#### Step 9
## Excute UNet Model
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

#### Step 10
## Training the model
train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)


#### Step 11
## Save file name
hdf5FileName = 'unet-'+ str(calendar.timegm(time.gmtime())) + '.h5'
model.save_weights(hdf5FileName)

#### Step 12
## Uploading h5 File
folderName = 'U-NET HDF5 Result'
folders = drive.ListFile(
    {'q': "title='" + folderName + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
for folder in folders:
    if folder['title'] == folderName:
        model_file = drive.CreateFile({'parents': [{'id': folder['id']}]})
        model_file.SetContentFile(hdf5FileName)
        model_file.Upload()
        print('Uploaded ' + hdf5FileName + ' success')

#### Step 13
## Remove dataset 
# mypath = os.path.abspath('dataset')
# os.system('rm -rf %s' % mypath)

## Dataset for prediction
# x, y = valid_gen.__getitem__(1)
# result = model.predict(x)

# result = result > 0.5

