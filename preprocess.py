import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier

import pandas as pd
import numpy as np

from diag_utils import map_diagnos, map_diagnos_list

# prepare data

img2vec = Img2Vec(model='resnet-18')

data_dir = './dat'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# metadata_fields = ['benign_malignant', 'diagnosis']
# train_metadata = pd.read_csv(os.path.join(data_dir, 'train_metadata.csv'), skipinitialspace=True, usecols=metadata_fields )
# val_metadata = pd.read_csv(os.path.join(data_dir, 'val_metadata.csv'), skipinitialspace=True, usecols=metadata_fields)

# train_metadata  = list(zip(train_metadata.benign_malignant,    train_metadata.diagnosis))
# val_metadata    = list(zip(val_metadata.benign_malignant,      val_metadata.diagnosis))
# train_metadata  = map_diagnos_list(train_metadata)
# val_metadata    = map_diagnos_list(val_metadata)

# np.array(train_metadata, dtype=np.uint16).tofile('training_labels.bin')
# np.array(val_metadata, dtype=np.uint16).tofile('validation_labels.bin')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    i = 0
    print(f'--- inserting data from {dir_} ---')
    for img_path in os.listdir(dir_):
        img_path_ = os.path.join(dir_, img_path)
        img = Image.open(img_path_)

        img_features = img2vec.get_vec(img)
        features.append(img_features)

        # print(img_path)
        features_np = np.array(features)
        i += 1
        if i % 1000 == 0:
            print(f'{i} images processed')
    data[['training_data', 'validation_data'][j]] = np.array(features)
    
    data[['training_data', 'validation_data'][j]].tofile(['training_data_512.bin', 'validation_data_512.bin'][j])
    print(f"DATA exported to {['training_data_512.bin', 'validation_data_512.bin'][j]} successfully")
