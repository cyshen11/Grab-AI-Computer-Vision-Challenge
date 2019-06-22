import numpy as np 
import pandas as pd 
import os
from os import listdir
from keras.models import Model
from resnet import resnet18,load_trained_model
import scipy.io
from keras.preprocessing import image

image_shape = (260,504,3)
n_classes = 196

# Load cars_meta into dataFrame
meta_dict = scipy.io.loadmat('cars_meta.mat')
meta = pd.DataFrame.from_dict(meta_dict['class_names'])
meta = meta.stack()
meta_df = pd.DataFrame(meta, columns = ["Car Make"])

# Load model
model = resnet18(image_shape,n_classes)
weights_path = 'resnet34_val_acc_46.hdf5'
model.load_weights(weights_path)

test_dataset_path = input('\nEnter test dataset folder path:')
imgs = [f for f in listdir(test_dataset_path)]
df_imgs = pd.DataFrame(imgs, columns = ["Image"])

result = pd.DataFrame(columns=['Image','Prediction'],index=range(len(imgs)))

for i,img in enumerate(imgs):
    file = test_dataset_path + img
    test_img = image.load_img(file, target_size=(260,504))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    predict = model.predict(test_img)
    result['Image'][i] = img
    result['Prediction'][i] = meta_df['Car Make'][np.argmax(predict,axis=1)[0]]

print(result)